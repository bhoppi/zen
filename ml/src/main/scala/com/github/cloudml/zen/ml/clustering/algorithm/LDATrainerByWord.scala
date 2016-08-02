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

import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg.{sum, DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.clustering.LDAPrecalc._
import com.github.cloudml.zen.ml.util.Concurrent._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.concurrent.Future


abstract class LDATrainerByWord(numTopics: Int, numThreads: Int)
  extends LDATrainer(numTopics, numThreads) {
  override def isByDoc: Boolean = false

  override def initEdgePartition(ep: EdgePartition[TA, _]): EdgePartition[TA, Int] = {
    val totalSize = ep.size
    val srcSize = ep.indexSize
    val lcSrcIds = ep.localSrcIds
    val zeros = new Array[Int](ep.vertexAttrs.length)
    val srcInfos = new Array[(Int, Int, Int)](srcSize)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator.zipWithIndex) { case ((_, startPos), ii) => withFuture {
      val si = lcSrcIds(startPos)
      var pos = startPos
      while (pos < totalSize && lcSrcIds(pos) == si) {
        pos += 1
      }
      srcInfos(ii) = (si, startPos, pos)
    }}
    withAwaitReadyAndClose(all)

    val newLcSrcIds = srcInfos.toSeq.sorted.flatMap(t => Iterator(t._1, t._2, t._3)).toArray
    new EdgePartition(newLcSrcIds, ep.localDstIds, ep.data, null, ep.global2local, ep.local2global, zeros, None)
  }

  override def countPartition(ep: EdgePartition[TA, Int]): Iterator[NvkPair] = {
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val l2g = ep.local2global
    val useds = ep.vertexAttrs
    val data = ep.data
    val vertSize = useds.length
    val results = new Array[NvkPair](vertSize)

    implicit val es = initExecutionContext(numThreads)
    val all0 = Future.traverse(Range(0, numThreads).iterator) { thid => withFuture {
      var i = thid
      while (i < vertSize) {
        val vid = l2g(i)
        val used = useds(i)
        val counter: Nvk = if (isTermId(vid) && used >= dscp) {
          new BDV(new Array[Count](numTopics))
        } else {
          val len = math.max(used >>> 1, 2)
          new BSV(new Array[Int](len), new Array[Count](len), 0, numTopics)
        }
        results(i) = (vid, counter)
        i += numThreads
      }
    }}
    withAwaitReady(all0)

    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
      val si = lcSrcIds(lsi)
      val startPos = lcSrcIds(lsi + 1)
      val endPos = lcSrcIds(lsi + 2)
      val termTopics = results(si)._2
      var pos = startPos
      while (pos < endPos) {
        val docTopics = results(lcDstIds(pos))._2
        val topic = data(pos)
        termTopics(topic) += 1
        docTopics.synchronized {
          docTopics(topic) += 1
        }
        pos += 1
      }
      termTopics match {
        case v: BDV[Count] =>
          val used = v.data.count(_ > 0)
          if (used < dscp) {
            results(si) = (l2g(si), toBSV(v, used))
          }
        case _ =>
      }
    }}
    withAwaitReadyAndClose(all)

    results.iterator
  }

  override def perplexPartition(globalCountersBc: Broadcast[LDAGlobalCounters],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (ep: EdgePartition[TA, Nvk]): (Double, Double, Double) = {
    val topicCounters = globalCountersBc.value
    val alphaSum = alpha * numTopics
    val betaSum = beta * numTerms
    val alphaRatio = calc_alphaRatio(numTopics, numTokens, alphaAS, alphaSum)
    val alphaks = calc_alphaks(topicCounters, alphaAS, alphaRatio)
    val denoms = calc_denoms(topicCounters, numTopics, betaSum)
    val alphak_denoms = calc_alphak_denoms(denoms, alphaAS, betaSum, alphaRatio)

    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val vertSize = vattrs.length
    val docNorms = new Array[Double](vertSize)
    @volatile var llhs = 0.0
    @volatile var wllhs = 0.0
    @volatile var dllhs = 0.0
    val abDenseSum = sum_abDense(alphak_denoms, beta)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
      val si = lcSrcIds(lsi)
      val startPos = lcSrcIds(lsi + 1)
      val endPos = lcSrcIds(lsi + 2)
      val termTopics = vattrs(si)
      val waSparseSum = sum_waSparse(alphak_denoms, termTopics)
      val sum12 = abDenseSum + waSparseSum
      var llhs_th = 0.0
      var wllhs_th = 0.0
      var dllhs_th = 0.0
      val denseTermTopics = toBDV(termTopics)
      var pos = startPos
      while (pos < endPos) {
        val di = lcDstIds(pos)
        val docTopics = vattrs(di).asInstanceOf[Ndk]
        var docNorm = docNorms(di)
        if (docNorm == 0.0) {
          docNorm = 1.0 / (sum(docTopics) + alphaSum)
          docNorms(di) = docNorm
        }
        val dwbSparseSum = sum_dwbSparse(denoms, denseTermTopics, docTopics, beta)
        llhs_th += Math.log((sum12 + dwbSparseSum) * docNorm)
        val topic = data(pos)
        wllhs_th += Math.log((denseTermTopics(topic) + beta) * denoms(topic))
        dllhs_th += Math.log((docTopics(topic) + alphaks(topic)) * docNorm)
        pos += 1
      }
      llhs += llhs_th
      wllhs += wllhs_th
      dllhs += dllhs_th
    }}
    withAwaitReadyAndClose(all)

    (llhs, wllhs, dllhs)
  }

  def sum_waSparse(alphak_denoms: BDV[Double],
    termTopics: Nwk): Double = termTopics match {
    case v: BDV[Count] =>
      var sum = 0.0
      var i = 0
      while (i < numTopics) {
        val cnt = v(i)
        if (cnt > 0) {
          sum += alphak_denoms(i) * cnt
        }
        i += 1
      }
      sum
    case v: BSV[Count] =>
      val used = v.used
      val index = v.index
      val data = v.data
      var sum = 0.0
      var i = 0
      while (i < used) {
        sum += alphak_denoms(index(i)) * data(i)
        i += 1
      }
      sum
  }

  def sum_dwbSparse(denoms: BDV[Double],
    denseTermTopics: BDV[Count],
    docTopics: Ndk,
    beta: Double): Double = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var sum = 0.0
    var i = 0
    while (i < used) {
      val topic = index(i)
      sum += (denseTermTopics(topic) + beta) * data(i) * denoms(topic)
      i += 1
    }
    sum
  }

  def sum_dwbSparse_wOpt(termBeta_denoms: BDV[Double],
    docTopics: Ndk): Double = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var sum = 0.0
    var i = 0
    while (i < used) {
      sum += termBeta_denoms(index(i)) * data(i)
      i += 1
    }
    sum
  }

  override def docOfWordsPartition(candWordsBc: Broadcast[Set[Int]])
    (ep: EdgePartition[TA, _]): Iterator[(Int, mutable.ArrayBuffer[Long])] = {
    val candWords = candWordsBc.value
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val l2g = ep.local2global
    val dwp = new ConcurrentLinkedQueue[(Int, mutable.ArrayBuffer[Long])]

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(lcSrcIds.indices.by(3).iterator) { lsi => withFuture {
      val si = lcSrcIds(lsi)
      val wid = l2g(si).toInt
      if (candWords.contains(wid)) {
        val startPos = lcSrcIds(lsi + 1)
        val endPos = lcSrcIds(lsi + 2)
        val docs = new mutable.ArrayBuffer[Long]
        lcDstIds.toSeq.slice(startPos, endPos).distinct.map(l2g(_)).copyToBuffer(docs)
        dwp.add((wid, docs))
      }
    }}
    withAwaitReadyAndClose(all)

    dwp.toSeq.iterator
  }
}
