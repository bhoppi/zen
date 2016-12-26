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

import java.util.concurrent.atomic.AtomicIntegerArray

import breeze.linalg._
import breeze.numerics._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.concurrent.TrieMap


class GLDATrainer(numTopics: Int, numGroups: Int, numThreads: Int)
  extends Serializable {
  def SampleNGroup(dataBlocks: RDD[(Int, DataBlock)],
    shippeds: RDD[(Int, ShippedAttrsBlock)],
    globalVarsBc: Broadcast[GlobalVars],
    params: HyperParams,
    seed: Int,
    sampIter: Int,
    burninIter: Int): RDD[(Int, DataBlock)] = {
    val newSeed = (seed + sampIter) * dataBlocks.partitions.length
    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = dataBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    isoRDD.zipPartitions(shippeds, preservesPartitioning=true) { (dataIter, shpsIter) =>
      dataIter.map { case (pid, DataBlock(termRecs, docRecs)) =>
        val totalDocSize = docRecs.length
        implicit val es = newExecutionContext(numThreads)

        // Stage 1: assign all termTopics
        var startTime = System.nanoTime
        val termVecs = new TrieMap[Int, CompressedVector]()
        parallelized_foreachElement[(Int, ShippedAttrsBlock)](shpsIter, numThreads, (shp, _) => {
          val (_, ShippedAttrsBlock(termIds, termAttrs)) = shp
          termIds.iterator.zip(termAttrs.iterator).foreach { case (termId, termAttr) =>
            termVecs(termId) = termAttr
          }
        })
        var endTime = System.nanoTime
        var elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): Assigning all termTopics takes: ${elapsed}s.")

        // Stage 2: calculate all docTopics, docLen, docGrp
        startTime = System.nanoTime
        val docResults = new Array[(SparseVector[Int], Int, Int)](totalDocSize)
        parallelized_foreachSplit(totalDocSize, numThreads, (ds, dn, _) => {
          var di = ds
          while (di < dn) {
            val DocRec(_, docGrp, docData) = docRecs(di)
            val docTopics = SparseVector.zeros[Int](numTopics)
            var p = 0
            while (p < docData.length) {
              var ind = docData(p)
              if (ind >= 0) {
                docTopics(docData(p + 1)) += 1
                p += 2
              } else {
                p += 2
                while (ind < 0) {
                  docTopics(docData(p)) += 1
                  p += 1
                  ind += 1
                }
              }
            }
            docTopics.compact()
            docResults(di) = (docTopics, sum(docTopics), docGrp.value & 0xFFFF)
            di += 1
          }
        })
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): calculate all docTopics, docLen, docGrp takes: ${elapsed}s.")

        // Stage 3: token sampling
        startTime = System.nanoTime
        val eta = params.eta
        val mu = params.mu
        val gens = Array.tabulate(numThreads)(thid => new XORShiftRandom((newSeed + pid) * numThreads + thid))
        val decomps = Array.fill(numThreads)(new BVDecompressor(numTopics))
        val cdfDists = Array.fill(numThreads)(new CumulativeDist[Double]().reset(numTopics))
        val GlobalVars(piGK, nK, dG) = globalVarsBc.value
        val megDists = resetDists_megDenses(piGK, eta, mu)
        parallelized_foreachBatch[TermRec](termRecs.iterator, numThreads, 100, (batch, _, thid) => {
          val gen = gens(thid)
          val decomp = decomps(thid)
          val cdfDist = cdfDists(thid)
          val totalSamp = new CompositeSampler()
          batch.foreach { termRec =>
            val TermRec(termId, termData) = termRec
            val termTopics = decomp.CV2BV(termVecs(termId))
            val denseTermTopics = getDensed(termTopics)
            val tegDists = new Array[AliasTable[Double]](numGroups)
            val resetDist_tegSparse_f = resetDist_tegSparse(piGK, nK, termTopics, denseTermTopics, eta) _
            val resetDist_dtmSparse_f = resetDist_dtmSparse(nK, denseTermTopics) _
            val resetDist_dtmSparse_wAdj_f = resetDist_dtmSparse_wAdj(nK, denseTermTopics) _
            var p = 0
            while (p < termData.length) {
              val di = termData(p)
              var q = termData(p + 1)
              val docData = docRecs(di).docData
              val (docTopics, docLen, g) = docResults(di)
              val tegDist = {
                var dist = tegDists(g)
                if (dist == null) {
                  dist = resetDist_tegSparse_f(g)
                  tegDists(g) = dist
                }
                dist
              }
              var ind = docData(q)
              if (ind >= 0) {
                val topic = docData(q + 1)
                resetDist_dtmSparse_wAdj_f(cdfDist, docTopics, docLen, mu, topic)
                totalSamp.resetComponents(cdfDist, tegDist, megDists(g))
                docData(q + 1) = totalSamp.resampleRandom(gen, topic)
              } else {
                resetDist_dtmSparse_f(cdfDist, docTopics, docLen, mu)
                totalSamp.resetComponents(cdfDist, tegDist, megDists(g))
                q += 2
                while (ind < 0) {
                  docData(q) = totalSamp.resampleRandom(gen, docData(q))
                  q += 1
                  ind += 1
                }
              }
              p += 2
            }
          }
        })
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): Sampling tokens takes: ${elapsed}s.")

        // Stage 4: doc grouping
        startTime = System.nanoTime
        val priors = log(convert(dG, Double) :+= 1.0)
        parallelized_foreachSplit(totalDocSize, numThreads, (ds, dn, thid) => {
          val gen = gens(thid)
          val samp = new CumulativeDist[Double]().reset(numGroups)
          var di = ds
          while (di < dn) {
            val docGrp = docRecs(di).docGrp
            if (docGrp.value < 0x10000) {
              val (docTopics, docLen, _) = docResults(di)
              val llhs = priors.copy
              val used = docTopics.used
              val index = docTopics.index
              val data = docTopics.data
              var i = 0
              while (i < used) {
                val k = index(i)
                val ndk = data(i)
                var g = 0
                while (g < numGroups) {
                  val sgk = (piGK(g, k) * docLen).toDouble
                  llhs(g) += lgamma(sgk + ndk) - lgamma(sgk)
                  g += 1
                }
                i += 1
              }
              val ng = if (sampIter <= burninIter) {
                llhs :-= max(llhs)
                samp.resetDist(exp(llhs).iterator, numGroups).sampleRandom(gen)
              } else {
                argmax(llhs)
              }
              docGrp.value = ng
            }
            di += 1
          }
        }, closing=true)
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): Grouping docs takes: ${elapsed}s.")

        (pid, DataBlock(termRecs, docRecs))
      }
    }
  }

  def resetDists_megDenses(piGK: DenseMatrix[Float], eta: Double, mu: Double): Array[AliasTable[Double]] = {
    val egs = new Array[AliasTable[Double]](numGroups)
    Range(0, numGroups).par.foreach { g =>
      val probs = convert(piGK(g, ::).t, Double) :*= (eta * mu)
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
    mu: Double): CumulativeDist[Double] = {
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
      sum += (mu + denseTermTopics(k).toDouble / nK(k)) * data(i) / docLen
      cdf(i) = sum
      i += 1
    }
    dtm._space = index
    dtm.setResidualRate { k =>
      val a = 1.0 / (denseTermTopics(k) + mu * nK(k))
      val b = 1.0 / docTopics(k)
      a + b - a * b
    }
    dtm
  }

  def resetDist_dtmSparse_wAdj(nK: Array[Int],
    denseTermTopics: DenseVector[Int])(dtm: CumulativeDist[Double],
    docTopics: SparseVector[Int],
    docLen: Int,
    mu: Double,
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
        (mu + (denseTermTopics(k) - 1).toDouble / nK(k)) * (cnt - 1)
      } else {
        (mu + denseTermTopics(k).toDouble / nK(k)) * cnt
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
      parallelized_mapElement[TermRec, (Int, SparseVector[Int])](termRecs.iterator, numThreads, termRec => {
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
      }, closing=true)(newExecutionContext(numThreads))
    }).partitionBy(paraBlocks.partitioner.get)

    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = paraBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    isoRDD.zipPartitions(shippeds, preservesPartitioning=true)((paraIter, shpsIter) =>
      paraIter.map { case (pid, ParaBlock(routes, index, attrs)) =>
        val totalTermSize = attrs.length
        val results = new Array[Vector[Int]](totalTermSize)
        val marks = new AtomicIntegerArray(totalTermSize)
        implicit val es = newExecutionContext(numThreads)

        parallelized_foreachBatch[(Int, SparseVector[Int])](shpsIter, numThreads, numThreads * 5, (batch, _, _) => {
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

        parallelized_foreachSplit(totalTermSize, numThreads, (ts, tn, _) => {
          val comp = new BVCompressor(numTopics)
          var ti = ts
          while (ti < tn) {
            attrs(ti) = comp.BV2CV(results(ti))
            ti += 1
          }
        }, closing=true)

        (pid, ParaBlock(routes, index, attrs))
      }
    )
  }

  def collectGlobalVariables(dataBlocks: RDD[(Int, DataBlock)],
    params: HyperParams,
    numTerms: Int): GlobalVars = {
    type Pair = (DenseMatrix[Int], DenseVector[Long])
    val reducer: (Pair, Pair) => Pair = (a, b) => (a._1 :+= b._1, a._2 :+= b._2)
    val (nGK, dG) = dataBlocks.mapPartitions(_.map { dbp =>
      val docRecs = dbp._2.DocRecs

      parallelized_reduceSplit[Pair](docRecs.length, numThreads, (ds, dn) => {
        val nGKThrd = DenseMatrix.zeros[Int](numGroups, numTopics)
        val dGThrd = DenseVector.zeros[Long](numGroups)
        var di = ds
        while (di < dn) {
          val DocRec(_, docGrp, docData) = docRecs(di)
          val g = docGrp.value & 0xFFFF
          var p = 0
          while (p < docData.length) {
            var ind = docData(p)
            if (ind >= 0) {
              nGKThrd(g, docData(p + 1)) += 1
              p += 2
            } else {
              p += 2
              while (ind < 0) {
                nGKThrd(g, docData(p)) += 1
                p += 1
                ind += 1
              }
            }
          }
          dGThrd(g) += 1
          di += 1
        }
        (nGKThrd, dGThrd)
      }, reducer, closing=true)(newExecutionContext(numThreads))
    }).treeReduce(reducer)

    val nG = Range(0, numGroups).par.map(g => sum(convert(nGK(g, ::).t, Long))).toArray
    val nK = Range(0, numTopics).par.map(k => sum(nGK(::, k))).toArray
    val alpha = params.alpha
    val piGK = DenseMatrix.zeros[Float](numGroups, numTopics)
    Range(0, numGroups).par.foreach { g =>
      val piGKDenom = 1f / (nG(g) + alpha * numTopics)
      for (k <- 0 until numTopics) {
        piGK(g, k) = (nGK(g, k) + alpha) * piGKDenom
      }
    }
    GlobalVars(piGK, nK, dG)
  }

  def ShipParaBlocks(dataBlocks: RDD[(Int, DataBlock)],
    paraBlocks: RDD[(Int, ParaBlock)]): RDD[(Int, ShippedAttrsBlock)] = {
    paraBlocks.mapPartitions(_.flatMap { case (_, ParaBlock(routes, index, attrs)) =>
      parallelized_mapBatch[Int, (Int, ShippedAttrsBlock)](routes.indices.iterator, numThreads, pid => {
        val termIds = routes(pid)
        val termAttrs = termIds.map(termId => attrs(index(termId)))
        (pid, ShippedAttrsBlock(termIds, termAttrs))
      }, closing=true)(newExecutionContext(numThreads))
    }).partitionBy(dataBlocks.partitioner.get)
  }
}
