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

package com.github.cloudml.zen.ml.clustering.algorithm

import java.lang.ref.SoftReference
import java.util.Random
import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent.Future


class AliasLDA_MCEM(numTopics: Int, numThreads: Int)
  extends LDATrainerByDoc(numTopics: Int, numThreads: Int) {
  override def initEdgePartition(ep: EdgePartition[TA, _]): EdgePartition[TA, Int] = {
    val totalSize = ep.size
    val srcSize = ep.indexSize
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val zeros = new Array[Int](ep.vertexAttrs.length)
    val srcInfos = new Array[(Int, Int, Int)](srcSize)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => withFuture {
      val si = lcSrcIds(startPos)
      var anchor = startPos
      var anchorId = lcDstIds(anchor)
      var pos = startPos + 1
      while (pos < totalSize && lcSrcIds(pos) == si) {
        val lcDstId = lcDstIds(pos)
        if (lcDstId != anchorId) {
          val numLink = pos - anchor
          if (numLink > 1) {
            lcDstIds(anchor) = -numLink
          }
          anchor = pos
          anchorId = lcDstId
        }
        pos += 1
      }
      val numLink = pos - anchor
      if (numLink > 1) {
        lcDstIds(anchor) = -numLink
      }
      srcInfos(ii) = (si, startPos, pos)
    }}
    withAwaitReadyAndClose(all)

    val newLcSrcIds = srcInfos.toSeq.sorted.flatMap(t => Iterator(t._1, t._2, t._3)).toArray
    new EdgePartition(newLcSrcIds, lcDstIds, ep.data, null, ep.global2local, ep.local2global, zeros, None)
  }

  override def samplePartition(numPartitions: Int,
    sampIter: Int,
    seed: Int,
    topicCounters: BDV[Count],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (pid: Int, ep: EdgePartition[TA, Nvk]): EdgePartition[TA, Int] = {
    val alphaSum = alpha * numTopics
    val betaSum = beta * numTerms
    val alphaRatio = calc_alphaRatio(alphaSum, numTokens, alphaAS)
    val denoms = calc_denoms(topicCounters, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)

    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val vertSize = vattrs.length
    val useds = new Array[Int](vertSize)
    val termCache = new Array[SoftReference[AliasTable[Double]]](vertSize)

    val global = new AliasTable[Double]
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val docDists = new Array[AliasTable[Double]](numThreads)
    resetDist_abDense(global, alphak_denoms, beta)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator) { case (_, startPos) => withFuture {
      val thid = thq.poll() - 1
      try {
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          docDists(thid) = new AliasTable[Double].reset(numTopics)
        }
        val docDist = docDists(thid)

        val si = lcSrcIds(startPos)
        val docTopics = vattrs(si).asInstanceOf[Ndk]
        useds(si) = docTopics.activeSize
        var pos = startPos
        while (pos < totalSize && lcSrcIds(pos) == si) {
          var ind = lcDstIds(pos)
          if (ind >= 0) {
            val di = ind
            val termTopics = vattrs(di)
            useds(di) = termTopics.activeSize
            val termDist = waSparseCached(termCache, di, needRefresh=false).getOrElse {
              resetCache_waSparse(termCache, di, alphak_denoms, termTopics)
            }
            val topic = data(pos)
            resetDist_dwbSparse_wAdjust(docDist, denoms, termTopics, docTopics, beta, topic)
            data(pos) = tokenSampling(gen, global, termDist, docDist, termTopics, topic)
            pos += 1
          } else {
            val di = lcDstIds(pos + 1)
            val termTopics = vattrs(di)
            useds(di) = termTopics.activeSize
            val termDist = waSparseCached(termCache, di, needRefresh=false).getOrElse {
              resetCache_waSparse(termCache, di, alphak_denoms, termTopics)
            }
            resetDist_dwbSparse(docDist, denoms, termTopics, docTopics, beta)
            while (ind < 0) {
              val topic = data(pos)
              data(pos) = tokenResampling(gen, global, termDist, docDist, termTopics, docTopics, topic, beta)
              pos += 1
              ind += 1
            }
          }
        }
      } finally {
        thq.add(thid + 1)
      }
    }}
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(useds)
  }

  def tokenSampling(gen: Random,
    ab: AliasTable[Double],
    wa: AliasTable[Double],
    dwb: AliasTable[Double],
    termTopics: Nwk,
    curTopic: Int): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.sampleFrom(genSum, gen)
    } else if (genSum < sum23) {
      wa.resampleFrom(genSum - dwbSum, gen, curTopic, 1.0 / termTopics(curTopic))
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  def tokenResampling(gen: Random,
    ab: AliasTable[Double],
    wa: AliasTable[Double],
    dwb: AliasTable[Double],
    termTopics: Nwk,
    docTopics: Ndk,
    curTopic: Int,
    beta: Double): Int = {
    val dwbSum = dwb.norm
    val sum23 = dwbSum + wa.norm
    val distSum = sum23 + ab.norm
    val genSum = gen.nextDouble() * distSum
    if (genSum < dwbSum) {
      dwb.resampleFrom(genSum, gen, curTopic, {
        val a = 1.0 / (termTopics(curTopic) + beta)
        val b = 1.0 / docTopics(curTopic)
        a + b - a * b
      })
    } else if (genSum < sum23) {
      wa.resampleFrom(genSum - dwbSum, gen, curTopic, 1.0 / termTopics(curTopic))
    } else {
      ab.sampleFrom(genSum - sum23, gen)
    }
  }

  def resetDist_abDense(ab: AliasTable[Double],
    alphak_denoms: BDV[Double],
    beta: Double): AliasTable[Double] = {
    val probs = alphak_denoms.copy :*= beta
    ab.resetDist(probs.data, null, probs.length)
  }

  def resetDist_dwbSparse(dwb: AliasTable[Double],
    denoms: BDV[Double],
    termTopics: Nwk,
    docTopics: Ndk,
    beta: Double): AliasTable[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    val probs = new Array[Double](used)
    var i = 0
    while (i < used) {
      val topic = index(i)
      probs(i) = (termTopics(topic) + beta) * data(i) * denoms(topic)
      i += 1
    }
    dwb.resetDist(probs, index, used)
  }

  def resetDist_dwbSparse_wAdjust(dwb: AliasTable[Double],
    denoms: BDV[Double],
    termTopics: Nwk,
    docTopics: Ndk,
    beta: Double,
    curTopic: Int): AliasTable[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    val probs = new Array[Double](used)
    var i = 0
    while (i < used) {
      val topic = index(i)
      val docCnt = data(i)
      val termBeta = termTopics(topic) + beta
      val numer = if (topic == curTopic) {
        (termBeta - 1.0) * (docCnt - 1)
      } else {
        termBeta * docCnt
      }
      probs(i) = numer * denoms(topic)
      i += 1
    }
    dwb.resetDist(probs, index, used)
  }

  def waSparseCached(cache: Array[SoftReference[AliasTable[Double]]],
    ci: Int,
    needRefresh: => Boolean): Option[AliasTable[Double]] = {
    val termCache = cache(ci)
    if (termCache == null || termCache.get == null || needRefresh) {
      None
    } else {
      Some(termCache.get)
    }
  }

  def resetCache_waSparse(cache: Array[SoftReference[AliasTable[Double]]],
    ci: Int,
    alphak_denoms: BDV[Double],
    termTopics: Nwk): AliasTable[Double] = {
    val table = new AliasTable[Double]
    termTopics match {
      case v: BDV[Count] =>
        val probs = new Array[Double](numTopics)
        val space = new Array[Int](numTopics)
        var psize = 0
        var i = 0
        while (i < numTopics) {
          val cnt = v(i)
          if (cnt > 0) {
            probs(psize) = alphak_denoms(i) * cnt
            space(psize) = i
            psize += 1
          }
          i += 1
        }
        table.resetDist(probs, space, psize)
      case v: BSV[Count] =>
        val used = v.used
        val index = v.index
        val data = v.data
        val probs = new Array[Double](used)
        var i = 0
        while (i < used) {
          probs(i) = alphak_denoms(index(i)) * data(i)
          i += 1
        }
        table.resetDist(probs, index, used)
    }
    cache(ci) = new SoftReference(table)
    table
  }
}
