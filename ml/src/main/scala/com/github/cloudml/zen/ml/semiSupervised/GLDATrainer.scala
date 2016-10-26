/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.cloudml.zen.ml.semiSupervised

import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicIntegerArray

import breeze.linalg._
import breeze.numerics._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.collection.concurrent.TrieMap
import scala.concurrent.Future


class GLDATrainer(numTopics: Int, numGroups: Int, numThreads: Int)
  extends Serializable {
  def SampleNGroup(dataBlocks: RDD[(Int, DataBlock)],
    shippeds: RDD[(Int, ShippedAttrsBlock)],
    globalVarsBc: Broadcast[GlobalVars],
    params: HyperParams,
    seed: Int,
    sampIter: Int,
    burninIter: Int): RDD[(Int, DataBlock)] = {
    val newSeed = (seed + sampIter) * dataBlocks.getNumPartitions
    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = dataBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    isoRDD.zipPartitions(shippeds, preservesPartitioning=true) { (dataIter, shpsIter) =>
      val GlobalVars(piGK, sigGW, nK, dG) = globalVarsBc.value
      dataIter.map { case (pid, DataBlock(termRecs, docRecs)) =>
        // Stage 1: assign all termTopics
        var startTime = System.nanoTime
        val termVecs = new TrieMap[Int, CompressedVector]()
        implicit val es = initExecutionContext(numThreads)
        val allAssign = shpsIter.map(shp => withFuture {
          val (_, ShippedAttrsBlock(termIds, termAttrs)) = shp
          termIds.iterator.zip(termAttrs.iterator).foreach { case (termId, termAttr) =>
            termVecs(termId) = termAttr
          }
        })
        withAwaitReady(Future.sequence(allAssign))
        var endTime = System.nanoTime
        var elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): Assigning all termTopics takes: ${elapsed}s.")

        // Stage 2: calculate all docTopics, docLen, docGrp
        startTime = System.nanoTime
        val totalDocSize = docRecs.length
        val docResults = new Array[(SparseVector[Int], Int, Int)](totalDocSize)
        val sizePerThrd = {
          val npt = totalDocSize / numThreads
          if (npt * numThreads == totalDocSize) npt else npt + 1
        }
        val allDocVecs = Range(0, numThreads).map(thid => withFuture {
          val posN = math.min(sizePerThrd * (thid + 1), totalDocSize)
          var pos = sizePerThrd * thid
          while (pos < posN) {
            val DocRec(_, docGrp, docData) = docRecs(pos)
            val docTopics = SparseVector.zeros[Int](numTopics)
            var i = 0
            while (i < docData.length) {
              var ind = docData(i)
              if (ind >= 0) {
                docTopics(docData(i + 1)) += 1
                i += 2
              } else {
                i += 2
                while (ind < 0) {
                  docTopics(docData(i)) += 1
                  i += 1
                  ind += 1
                }
              }
            }
            docTopics.compact()
            docResults(pos) = (docTopics, sum(docTopics), docGrp.value & 0xFFFF)
            pos += 1
          }
        })
        withAwaitReady(Future.sequence(allDocVecs))
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): calculate all docTopics, docLen, docGrp takes: ${elapsed}s.")

        // Stage 3: token sampling
        startTime = System.nanoTime
        val eta = params.eta
        val mu = params.mu
        val thq = new ConcurrentLinkedQueue(1 to numThreads)
        val gens = Array.tabulate(numThreads)(thid => new XORShiftRandom((newSeed + pid) * numThreads + thid))
        val decomps = Array.fill(numThreads)(new BVDecompressor(numTopics))
        val cdfDists = Array.fill(numThreads)(new CumulativeDist[Double]().reset(numTopics))
        val egDists = resetDists_egDenses(piGK, eta)
        val allSampling = termRecs.iterator.map(termRec => withFuture {
          val thid = thq.poll() - 1
          try {
            val gen = gens(thid)
            val decomp = decomps(thid)
            val cdfDist = cdfDists(thid)
            val scaleSamp = new ScalingSampler()
            val totalSamp = new CompositeSampler()

            val TermRec(termId, termData) = termRec
            val termTopics = decomp.CV2BV(termVecs(termId))
            val denseTermTopics = getDensed(termTopics)
            val tegDists = new Array[AliasTable[Double]](numGroups)
            val resetDist_tegSparse_f = resetDist_tegSparse(piGK, nK, termTopics, denseTermTopics, eta) _
            val resetDist_dtmSparse_f = resetDist_dtmSparse(nK, denseTermTopics) _
            val resetDist_dtmSparse_wAdj_f = resetDist_dtmSparse_wAdj(nK, denseTermTopics) _
            var i = 0
            while (i < termData.length) {
              val docPos = termData(i)
              var docI = termData(i + 1)
              val docData = docRecs(docPos).docData
              val (docTopics, docLen, g) = docResults(docPos)
              val tegDist = {
                var dist = tegDists(g)
                if (dist == null) {
                  dist = resetDist_tegSparse_f(g)
                  tegDists(g) = dist
                }
                dist
              }
              val muSig = mu * sigGW(g, termId)
              scaleSamp.resetScaling(muSig, egDists(g))
              var ind = docData(docI)
              if (ind >= 0) {
                val topic = docData(docI + 1)
                resetDist_dtmSparse_wAdj_f(cdfDist, docTopics, docLen, muSig, topic)
                totalSamp.resetComponents(cdfDist, tegDist, scaleSamp)
                docData(docI + 1) = totalSamp.resampleRandom(gen, topic)
              } else {
                resetDist_dtmSparse_f(cdfDist, docTopics, docLen, muSig)
                totalSamp.resetComponents(cdfDist, tegDist, scaleSamp)
                docI += 2
                while (ind < 0) {
                  docData(docI) = totalSamp.resampleRandom(gen, docData(docI))
                  docI += 1
                  ind += 1
                }
              }
              i += 2
            }
          } finally {
            thq.add(thid + 1)
          }
        })
        withAwaitReady(Future.sequence(allSampling))
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): Sampling tokens takes: ${elapsed}s.")

        // Stage 4: doc grouping
        startTime = System.nanoTime
        val samps = Array.fill(numThreads)(new CumulativeDist[Float]().reset(numGroups))
        val priors = log(convert(dG, Float) :+ 1f)
        val lnPiGK = log(piGK)
        val lnSigGW = log(sigGW)
        val allGrouping = Range(0, numThreads).map(thid => Future {
          val thid = thq.poll() - 1
          try {
            val gen = gens(thid)
            val samp = samps(thid)
            val posN = math.min(sizePerThrd * (thid + 1), totalDocSize)
            var pos = sizePerThrd * thid
            while (pos < posN) {
              val docGrp = docRecs(pos).docGrp
              if (docGrp.value < 0x10000) {
                val docData = docRecs(pos).docData
                val llhs = priors.copy
                var i = 0
                while (i < docData.length) {
                  var ind = docData(i)
                  if (ind >= 0) {
                    llhs :+= lnSigGW(::, ind)
                    llhs :+= lnPiGK(::, docData(i + 1))
                    i += 2
                  } else {
                    llhs :+= lnSigGW(::, docData(i + 1)) :* -ind.toFloat
                    i += 2
                    while (ind < 0) {
                      llhs :+= lnPiGK(::, docData(i))
                      i += 1
                      ind += 1
                    }
                  }
                }
                val ng = if (sampIter <= burninIter) {
                  llhs :-= max(llhs)
                  samp.resetDist(exp(llhs).iterator, numGroups).sampleRandom(gen)
                } else {
                  argmax(llhs)
                }
                docGrp.value = ng
              }
              pos += 1
            }
          } finally {
            thq.add(thid + 1)
          }
        })
        withAwaitReadyAndClose(Future.sequence(allGrouping))
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): Grouping docs takes: ${elapsed}s.")

        (pid, DataBlock(termRecs, docRecs))
      }
    }
  }

  def resetDists_egDenses(piGK: DenseMatrix[Float], eta: Double): Array[AliasTable[Double]] = {
    val egs = new Array[AliasTable[Double]](numGroups)
    Range(0, numGroups).par.foreach { g =>
      val probs = convert(piGK(g, ::).t, Double) :*= eta
      egs(g) = new AliasTable[Double]().resetDist(probs.data, null, numTopics)
    }
    egs
  }

  def resetDist_tegSparse(piGK: DenseMatrix[Float],
    nK: Array[Int],
    termTopics: Vector[Int],
    denseTermTopics: DenseVector[Int],
    eta: Double)(g: Int): AliasTable[Double] = {
    val tegDist = new AliasTable[Double]()
    termTopics match {
      case v: DenseVector[Int] =>
        val probs = new Array[Double](numTopics)
        val space = new Array[Int](numTopics)
        var psize = 0
        var k = 0
        while (k < numTopics) {
          val cnt = v(k)
          if (cnt > 0) {
            probs(psize) = eta * piGK(g, k) * cnt / nK(k)
            space(psize) = k
            psize += 1
          }
          k += 1
        }
        tegDist.resetDist(probs, space, psize)
      case v: SparseVector[Int] =>
        val used = v.used
        val index = v.index
        val data = v.data
        val probs = new Array[Double](used)
        var i = 0
        while (i < used) {
          val k = index(i)
          probs(i) = eta * piGK(g, k) * data(i) / nK(k)
          i += 1
        }
        tegDist.resetDist(probs, index, used)
    }
    tegDist.setResidualRate(1.0 / denseTermTopics(_))
    tegDist
  }

  def resetDist_dtmSparse(nK: Array[Int],
    denseTermTopics: DenseVector[Int])(dtm: CumulativeDist[Double],
    docTopics: SparseVector[Int],
    docLen: Int,
    muSig: Double): CumulativeDist[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    // DANGER operations for performance
    dtm._used = used
    val cdf = dtm._cdf
    var sum = 0.0
    var i = 0
    while (i < used) {
      val k = index(i)
      sum += (muSig + denseTermTopics(k).toDouble / nK(k)) * data(i) / docLen
      cdf(i) = sum
      i += 1
    }
    dtm._space = index
    dtm.setResidualRate { k =>
      val a = 1.0 / (denseTermTopics(k) + muSig * nK(k))
      val b = 1.0 / docTopics(k)
      a + b - a * b
    }
    dtm
  }

  def resetDist_dtmSparse_wAdj(nK: Array[Int],
    denseTermTopics: DenseVector[Int])(dtm: CumulativeDist[Double],
    docTopics: SparseVector[Int],
    docLen: Int,
    muSig: Double,
    curTopic: Int): CumulativeDist[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    // DANGER operations for performance
    dtm._used = used
    val cdf = dtm._cdf
    var sum = 0.0
    var i = 0
    while (i < used) {
      val k = index(i)
      val cnt = data(i)
      val prob = if (k == curTopic) {
        (muSig + (denseTermTopics(k) - 1).toDouble / nK(k)) * (cnt - 1)
      } else {
        (muSig + denseTermTopics(k).toDouble / nK(k)) * cnt
      }
      sum += prob / docLen
      cdf(i) = sum
      i += 1
    }
    dtm._space = index
    dtm.unsetResidualRate()
    dtm
  }

  def updateParaBlocks(dataBlocks: RDD[(Int, DataBlock)],
    paraBlocks: RDD[(Int, ParaBlock)]): RDD[(Int, ParaBlock)] = {
    val dscp = numTopics >>> 3
    val shippeds = dataBlocks.mapPartitions(_.flatMap { case (_, DataBlock(termRecs, docRecs)) =>
      implicit val es = initExecutionContext(numThreads)
      val allAgg = termRecs.iterator.map(termRec => withFuture {
        val TermRec(termId, termData) = termRec
        val termTopics = SparseVector.zeros[Int](numTopics)
        var i = 0
        while (i < termData.length) {
          val docPos = termData(i)
          var docI = termData(i + 1)
          val docData = docRecs(docPos).docData
          var ind = docData(docI)
          if (ind >= 0) {
            termTopics(docData(docI + 1)) += 1
          } else {
            docI += 2
            while (ind < 0) {
              termTopics(docData(docI)) += 1
              docI += 1
              ind += 1
            }
          }
          i += 2
        }
        (termId, termTopics)
      })
      withAwaitResultAndClose(Future.sequence(allAgg))
    }).partitionBy(paraBlocks.partitioner.get)

    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = paraBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    isoRDD.zipPartitions(shippeds, preservesPartitioning=true)((paraIter, shpsIter) =>
      paraIter.map { case (pid, ParaBlock(routes, index, attrs)) =>
        val totalSize = attrs.length
        val results = new Array[Vector[Int]](totalSize)
        val marks = new AtomicIntegerArray(totalSize)

        implicit val es = initExecutionContext(numThreads)
        val allAgg = shpsIter.grouped(numThreads * 5).map(batch => withFuture {
          batch.foreach { case (termId, termTopics) =>
            val i = index(termId)
            if (marks.getAndDecrement(i) == 0) {
              results(i) = termTopics
            } else {
              while (marks.getAndSet(i, -1) <= 0) {}
              results(i) = results(i) match {
                case v: DenseVector[Int] => v :+= termTopics
                case v: SparseVector[Int] =>
                  v :+= termTopics
                  if (v.activeSize >= dscp) getDensed(v) else v
              }
            }
            marks.set(i, Int.MaxValue)
          }
        })
        withAwaitReady(Future.sequence(allAgg))

        val sizePerthrd = {
          val npt = totalSize / numThreads
          if (npt * numThreads == totalSize) npt else npt + 1
        }
        val allComp = Range(0, numThreads).map(thid => withFuture {
          val comp = new BVCompressor(numTopics)
          val posN = math.min(sizePerthrd * (thid + 1), totalSize)
          var pos = sizePerthrd * thid
          while (pos < posN) {
            attrs(pos) = comp.BV2CV(results(pos))
            pos += 1
          }
        })
        withAwaitResultAndClose(Future.sequence(allComp))

        (pid, ParaBlock(routes, index, attrs))
      }
    )
  }

  def collectGlobalVariables(dataBlocks: RDD[(Int, DataBlock)],
    params: HyperParams,
    numTerms: Int): GlobalVars = {
    val (nGK, nGW, dG) = dataBlocks.mapPartitions(_.map { case (_, DataBlock(termRecs, docRecs)) =>
      val totalDocSize = docRecs.length

      val docGrps = new Array[Int](totalDocSize)
      val sizePerThrd = {
        val npt = totalDocSize / numThreads
        if (npt * numThreads == totalDocSize) npt else npt + 1
      }
      implicit val es = initExecutionContext(numThreads)
      val allNGK = Range(0, numThreads).map(thid => withFuture {
        val nGKThrd = DenseMatrix.zeros[Int](numGroups, numTopics)
        val dGThrd = DenseVector.zeros[Long](numGroups)
        val posN = math.min(sizePerThrd * (thid + 1), totalDocSize)
        var pos = sizePerThrd * thid
        while (pos < posN) {
          val DocRec(_, docGrp, docData) = docRecs(pos)
          val g = docGrp.value & 0xFFFF
          docGrps(pos) = g
          var i = 0
          while (i < docData.length) {
            var ind = docData(i)
            if (ind >= 0) {
              nGKThrd(g, docData(i + 1)) += 1
              i += 2
            } else {
              i += 2
              while (ind < 0) {
                nGKThrd(g, docData(i)) += 1
                i += 1
                ind += 1
              }
            }
          }
          dGThrd(g) += 1
          pos += 1
        }
        (nGKThrd, dGThrd)
      })
      val (nGKLc, dGLc) = withAwaitResult(Future.reduce(allNGK)((a, b) => (a._1 :+= b._1, a._2 :+= b._2)))

      val nGWLc = DenseMatrix.zeros[Int](numGroups, numTerms)
      val allNGW = termRecs.iterator.map(termRec => withFuture {
        val TermRec(termId, termData) = termRec
        var i = 0
        while (i < termData.length) {
          val docPos = termData(i)
          val docData = docRecs(docPos).docData
          val g = docGrps(docPos)
          val ind = docData(termData(i + 1))
          val termCnt = if (ind >= 0) 1 else -ind
          nGWLc(g, termId) += termCnt
          i += 2
        }
      })
      withAwaitReadyAndClose(Future.sequence(allNGW))
      (nGKLc, nGWLc, dGLc)
    }).collect().par.reduce((a, b) => (a._1 :+= b._1, a._2 :+= b._2, a._3 :+= b._3))

    val nG = Range(0, numGroups).par.map(g => sum(convert(nGK(g, ::).t, Long))).toArray
    val nK = Range(0, numTopics).par.map(k => sum(nGK(::, k))).toArray
    val alpha = params.alpha
    val beta = params.beta
    val piGK = DenseMatrix.zeros[Float](numGroups, numTopics)
    val sigGW = DenseMatrix.zeros[Float](numGroups, numTerms)
    Range(0, numGroups).par.foreach { g =>
      val piGKDenom = 1f / (nG(g) + alpha * numTopics)
      val sigGWDenom = 1f / (nG(g) + beta * numTerms)
      for (k <- 0 until numTopics) {
        val ngk = nGK(g, k)
        piGK(g, k) = (ngk + alpha) * piGKDenom
      }
      for (w <- 0 until numTerms) {
        sigGW(g, w) = (nGW(g, w) + beta) * sigGWDenom
      }
    }
    GlobalVars(piGK, sigGW, nK, dG)
  }

  def ShipParaBlocks(dataBlocks: RDD[(Int, DataBlock)],
    paraBlocks: RDD[(Int, ParaBlock)]): RDD[(Int, ShippedAttrsBlock)] = {
    paraBlocks.mapPartitions(_.flatMap { case (_, ParaBlock(routes, index, attrs)) =>
      implicit val es = initExecutionContext(numThreads)
      routes.indices.grouped(numThreads).flatMap { batch =>
        val all = Future.traverse(batch)(pid => withFuture {
          val termIds = routes(pid)
          val termAttrs = termIds.map(termId => attrs(index(termId)))
          (pid, ShippedAttrsBlock(termIds, termAttrs))
        })
        withAwaitResult(all)
      } ++ {
        closeExecutionContext(es)
        Iterator.empty
      }
    }).partitionBy(dataBlocks.partitioner.get)
  }
}
