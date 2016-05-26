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
import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.sampler._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.XORShiftRandom
import org.apache.spark.graphx2.impl.EdgePartition

import scala.collection.JavaConversions._
import scala.concurrent.Future


class AliasLDA(numTopics: Int, numThreads: Int)
  extends LDATrainerByDoc(numTopics: Int, numThreads: Int) {
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

    val totalSize = ep.size
    val lcSrcIds = ep.localSrcIds
    val lcDstIds = ep.localDstIds
    val vattrs = ep.vertexAttrs
    val data = ep.data
    val vertSize = vattrs.length
    val useds = new Array[Int](vertSize)
    val termCache = new Array[SoftReference[AliasTable[Double]]](vertSize)

    val global = new AliasTable[Double].addAuxDist(new FlatDist(isSparse=false))
    val thq = new ConcurrentLinkedQueue(1 to numThreads)
    val gens = new Array[XORShiftRandom](numThreads)
    val docDists = new Array[AliasTable[Double]](numThreads)
    val MHSamps = new Array[MetropolisHastings](numThreads)
    val compSamps = new Array[CompositeSampler](numThreads)
    resetDist_abDense(global, topicCounters, alphaAS, beta, betaSum, alphaRatio)
    val CGSCurry = tokenOrigProb(topicCounters, alphaAS, beta, betaSum, alphaRatio) _

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(ep.index.iterator) { case (_, startPos) => withFuture {
      val thid = thq.poll() - 1
      try {
        var gen = gens(thid)
        if (gen == null) {
          gen = new XORShiftRandom(((seed + sampIter) * numPartitions + pid) * numThreads + thid)
          gens(thid) = gen
          docDists(thid) = new AliasTable[Double].addAuxDist(new FlatDist(isSparse=true)).reset(numTopics)
          MHSamps(thid) = new MetropolisHastings
          compSamps(thid) = new CompositeSampler
        }
        val docDist = docDists(thid)
        val MHSamp = MHSamps(thid)
        val compSamp = compSamps(thid)

        val si = lcSrcIds(startPos)
        val docTopics = vattrs(si).asInstanceOf[Ndk]
        useds(si) = docTopics.activeSize
        var pos = startPos
        while (pos < totalSize && lcSrcIds(pos) == si) {
          val di = lcDstIds(pos)
          val termTopics = vattrs(di)
          useds(di) = termTopics.activeSize
          if (gen.nextDouble() < 1e-6) {
            resetDist_abDense(global, topicCounters, alphaAS, beta, betaSum, alphaRatio)
          }
          val termDist = waSparseCached(termCache, di, gen.nextDouble() < 1e-4).getOrElse {
            resetCache_waSparse(termCache, di, topicCounters, termTopics, alphaAS, betaSum, alphaRatio)
          }
          val topic = data(pos)
          resetDist_dwbSparse_wAdjust(docDist, topicCounters, termTopics, docTopics, beta, betaSum, topic)
          val CGSFunc = CGSCurry(termTopics, docTopics)(topic)
          compSamp.resetComponents(docDist, termDist, global)
          MHSamp.resetProb(CGSFunc, compSamp, topic)
          val newTopic = MHSamp.sampleRandom(gen)
          if (newTopic != topic) {
            data(pos) = newTopic
            topicCounters(topic) -= 1
            topicCounters(newTopic) += 1
            docTopics(topic) -= 1
            docTopics(newTopic) += 1
            termTopics.synchronized {
              termTopics(topic) -= 1
              termTopics(newTopic) += 1
            }
          }
          pos += 1
        }
      } finally {
        thq.add(thid + 1)
      }
    }}
    withAwaitReadyAndClose(all)

    ep.withVertexAttributes(useds)
  }

  def tokenOrigProb(topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double)
    (termTopics: Nwk, docTopics: Ndk)
    (curTopic: Int)(i: Int): Double = {
    val adjust = if (i == curTopic) -1 else 0
    val nk = topicCounters(i)
    val nwk = termTopics.synchronized(termTopics(i))
    val alphak = (nk + alphaAS) * alphaRatio
    (docTopics(i) + adjust + alphak) * (nwk + adjust + beta) / (nk + adjust + betaSum)
  }

  def resetDist_abDense(ab: AliasTable[Double],
    topicCounters: BDV[Count],
    alphaAS: Double,
    beta: Double,
    betaSum: Double,
    alphaRatio: Double): DiscreteSampler[Double] = {
    val probs = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      val nk = topicCounters(i)
      val alphak = (nk + alphaAS) * alphaRatio
      probs(i) = alphak * beta / (nk + betaSum)
      i += 1
    }
    ab.resetDist(probs, null, numTopics)
  }

  def resetDist_dwbSparse_wAdjust(dwb: AliasTable[Double],
    topicCounters: BDV[Count],
    termTopics: Nwk,
    docTopics: Ndk,
    beta: Double,
    betaSum: Double,
    curTopic: Int): AliasTable[Double] = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    val probs = new Array[Double](used)
    termTopics.synchronized {
      var i = 0
      while (i < used) {
        val topic = index(i)
        val adjust = if (topic == curTopic) -1 else 0
        probs(i) = (termTopics(topic) + adjust + beta) * (data(i) + adjust) /
          (topicCounters(topic) + adjust + betaSum)
        i += 1
      }
    }
    dwb.resetDist(probs, index.clone(), used)
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
    topicCounters: BDV[Count],
    termTopics: Nwk,
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double): AliasTable[Double] = {
    val tmpDocTopics = termTopics.synchronized(termTopics.copy)
    val table = new AliasTable[Double].addAuxDist(new FlatDist[Double](isSparse=true).reset(numTopics))
    tmpDocTopics match {
      case v: BDV[Count] =>
        val probs = new Array[Double](numTopics)
        val space = new Array[Int](numTopics)
        var psize = 0
        var i = 0
        while (i < numTopics) {
          val cnt = v(i)
          if (cnt > 0) {
            val nk = topicCounters(i)
            val alphak = (nk + alphaAS) * alphaRatio
            probs(psize) = alphak * cnt / (nk + betaSum)
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
          val nk = topicCounters(index(i))
          val alphak = (nk + alphaAS) * alphaRatio
          probs(i) = alphak * data(i) / (nk + betaSum)
          i += 1
        }
        table.resetDist(probs, index, used)
    }
    cache(ci) = new SoftReference(table)
    table
  }
}
