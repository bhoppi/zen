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
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.concurrent.TrieMap


class GLDATrainer(numTopics: Int, numThreads: Int)
  extends Serializable {
  def SampleNGroup(dataBlocks: RDD[(Int, DataBlock)],
    shippeds: RDD[(Int, ShippedAttrsBlock)],
    globalVarsBc: Broadcast[GlobalVars],
    numTerms: Int,
    params: HyperParams,
    groupContext: GroupContext,
    seed: Int,
    sampIter: Int): RDD[(Int, DataBlock)] = {
    val newSeed = (seed + sampIter) * dataBlocks.partitions.length
    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = dataBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    isoRDD.zipPartitions(shippeds, preservesPartitioning=true) { (dataIter, shpsIter) =>
      dataIter.map { case (pid, DataBlock(termRecs, docRecs)) =>
        val totalDocSize = docRecs.length
        implicit val pec = newParaExecutionContext(numThreads)

        // Stage 1: assign all termTopics
        var startTime = System.nanoTime
        val termVecs = new TrieMap[Int, CompressedVector]()
        parallelized_foreachElement[(Int, ShippedAttrsBlock)](shpsIter, shp => {
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
        val GlobalVars(piGK, nK, _) = globalVarsBc.value
        val docResults = new Array[(SparseVector[Int], Int, Int)](totalDocSize)
        parallelized_foreachSplit(totalDocSize, (ds, dn, _) => {
          val grouper = groupContext.getDocGrouper(piGK)
          for (di <- ds until dn) {
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
            val docLen = sum(docTopics)
            if (!groupContext.isBurnin && docGrp.value < 0x10000) {
              docGrp.value = grouper.getGrp(docTopics, docLen)
            }
            docResults(di) = (docTopics, docLen, docGrp.value & 0xFFFF)
          }
        })
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): calculate all docTopics, docLen, docGrp takes: ${elapsed}s.")

        // Stage 3: token sampling
        startTime = System.nanoTime
        val totalGroups = groupContext.totalGroups
        val etaSum = params.eta * numTopics
        val mu = params.mu
        val muSum = mu * numTerms
        val denoms = calc_denoms(nK, muSum)
        val gens = Array.tabulate(numThreads)(thid => new XORShiftRandom((newSeed + pid) * numThreads + thid))
        val decomps = Array.fill(numThreads)(new BVDecompressor(numTopics))
        val cdfDists = Array.fill(numThreads)(new CumulativeDist[Double]().reset(numTopics))
        val megDists = resetDists_megDenses(piGK, denoms, totalGroups, etaSum, mu)
        parallelized_foreachBatch[TermRec](termRecs.iterator, 100, (batch, thid) => {
          val gen = gens(thid)
          val decomp = decomps(thid)
          val cdfDist = cdfDists(thid)
          val totalSamp = new CompositeSampler()
          batch.foreach { termRec =>
            val TermRec(termId, termData) = termRec
            val termTopics = decomp.CV2BV(termVecs(termId))
            val denseTermTopics = getDensed(termTopics)
            val tegDists = new Array[AliasTable[Double]](totalGroups)
            val resetDist_tegSparse_f = resetDist_tegSparse(piGK, denoms, termTopics, denseTermTopics, etaSum) _
            val resetDist_dtmSparse_f = resetDist_dtmSparse(denoms, denseTermTopics) _
            val resetDist_dtmSparse_wAdj_f = resetDist_dtmSparse_wAdj(denoms, denseTermTopics) _
            var p = 0
            while (p < termData.length) {
              val di = termData(p)
              var q = termData(p + 1)
              val docData = docRecs(di).docData
              val (docTopics, _, g) = docResults(di)
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
                resetDist_dtmSparse_wAdj_f(cdfDist, docTopics, mu, topic)
                totalSamp.resetComponents(cdfDist, tegDist, megDists(g))
                docData(q + 1) = totalSamp.resampleRandom(gen, topic)
              } else {
                resetDist_dtmSparse_f(cdfDist, docTopics, mu)
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
        }, closing=true)
        endTime = System.nanoTime
        elapsed = (endTime - startTime) / 1e9
        println(s"(Iteration $sampIter): Sampling tokens takes: ${elapsed}s.")

        (pid, DataBlock(termRecs, docRecs))
      }
    }
  }

  def resetDists_megDenses(piGK: DenseMatrix[Float],
    denoms: Array[Double],
    totalGroups: Int,
    etaSum: Double,
    mu: Double): Array[AliasTable[Double]] = {
    val egs = new Array[AliasTable[Double]](totalGroups)
    Range(0, totalGroups).par.foreach { g =>
      val probs = new Array[Double](numTopics)
      var k = 0
      while (k < numTopics) {
        probs(k) = etaSum * mu * piGK(g, k) * denoms(k)
        k += 1
      }
      egs(g) = new AliasTable[Double]().resetDist(probs, null, numTopics)
    }
    egs
  }

  def resetDist_tegSparse(piGK: DenseMatrix[Float],
    denoms: Array[Double],
    termTopics: Vector[Int],
    denseTermTopics: DenseVector[Int],
    etaSum: Double)(g: Int): AliasTable[Double] = {
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
            probs(psize) = etaSum * piGK(g, k) * cnt * denoms(k)
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
          probs(i) = etaSum * piGK(g, k) * data(i) * denoms(k)
          i += 1
        }
        tegDist.resetDist(probs, index, used)
    }
    tegDist.setResidualRate(1.0 / denseTermTopics(_))
    tegDist
  }

  def resetDist_dtmSparse(denoms: Array[Double],
    denseTermTopics: DenseVector[Int])(dtm: CumulativeDist[Double],
    docTopics: SparseVector[Int],
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
      sum += (denseTermTopics(k) + mu) * data(i) * denoms(k)
      cdf(i) = sum
      i += 1
    }
    dtm._space = index
    dtm.setResidualRate { k =>
      val a = 1.0 / (denseTermTopics(k) + mu)
      val b = 1.0 / docTopics(k)
      a + b - a * b
    }
    dtm
  }

  def resetDist_dtmSparse_wAdj(denoms: Array[Double],
    denseTermTopics: DenseVector[Int])(dtm: CumulativeDist[Double],
    docTopics: SparseVector[Int],
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
        (denseTermTopics(k) - 1 + mu) * (cnt - 1)
      } else {
        (denseTermTopics(k) + mu) * cnt
      }
      sum += prob * denoms(k)
      cdf(i) = sum
      i += 1
    }
    dtm._space = index
    dtm.unsetResidualRate()
    dtm
  }

  def calc_denoms(nK: Array[Int], muSum: Double): Array[Double] = {
    val arr = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      arr(i) = 1.0 / (nK(i) + muSum)
      i += 1
    }
    arr
  }

  def updateParaBlocks(dataBlocks: RDD[(Int, DataBlock)],
    paraBlocks: RDD[(Int, ParaBlock)]): RDD[(Int, ParaBlock)] = {
    val dscp = numTopics >>> 3
    val shippeds = dataBlocks.mapPartitions(_.flatMap { case (_, DataBlock(termRecs, docRecs)) =>
      parallelized_mapElement[TermRec, (Int, SparseVector[Int])](termRecs.iterator, termRec => {
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
      }, closing=true)(newParaExecutionContext(numThreads))
    }).partitionBy(paraBlocks.partitioner.get)

    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = paraBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    isoRDD.zipPartitions(shippeds, preservesPartitioning=true)((paraIter, shpsIter) =>
      paraIter.map { case (pid, ParaBlock(routes, index, attrs)) =>
        val totalTermSize = attrs.length
        val results = new Array[Vector[Int]](totalTermSize)
        val marks = new AtomicIntegerArray(totalTermSize)
        implicit val pec = newParaExecutionContext(numThreads)

        parallelized_foreachBatch[(Int, SparseVector[Int])](shpsIter, numThreads * 5, (batch, _) => {
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

        parallelized_foreachSplit(totalTermSize, (ts, tn, _) => {
          val comp = new BVCompressor(numTopics)
          for (ti <- ts until tn) {
            attrs(ti) = comp.BV2CV(results(ti))
          }
        }, closing=true)

        (pid, ParaBlock(routes, index, attrs))
      }
    )
  }

  def collectGlobalVariables(dataBlocks: RDD[(Int, DataBlock)],
    params: HyperParams,
    totalGroups: Int): GlobalVars = {
    type Pair = (DenseMatrix[Int], DenseVector[Long])
    val reducer: (Pair, Pair) => Pair = (a, b) => (a._1 :+= b._1, a._2 :+= b._2)

    val (nGK, dG) = dataBlocks.mapPartitions(_.map { dbp =>
      val docRecs = dbp._2.DocRecs

      parallelized_reduceSplit[Pair](docRecs.length, (ds, dn, _) => {
        val nGKThrd = DenseMatrix.zeros[Int](totalGroups, numTopics)
        val dGThrd = DenseVector.zeros[Long](totalGroups)
        for (di <- ds until dn) {
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
        }
        (nGKThrd, dGThrd)
      }, reducer, closing=true)(newParaExecutionContext(numThreads))
    }).treeReduce(reducer)

    val nG = Range(0, totalGroups).par.map(g => sum(convert(nGK(g, ::).t, Long))).toArray
    val nK = Range(0, numTopics).par.map(k => sum(nGK(::, k))).toArray
    val alpha = params.alpha
    val piGK = DenseMatrix.zeros[Float](totalGroups, numTopics)
    Range(0, totalGroups).par.foreach { g =>
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
      parallelized_mapBatch[Int, (Int, ShippedAttrsBlock)](routes.indices.iterator, pid => {
        val termIds = routes(pid)
        val termAttrs = termIds.map(termId => attrs(index(termId)))
        (pid, ShippedAttrsBlock(termIds, termAttrs))
      }, closing=true)(newParaExecutionContext(numThreads))
    }).partitionBy(dataBlocks.partitioner.get)
  }
}
