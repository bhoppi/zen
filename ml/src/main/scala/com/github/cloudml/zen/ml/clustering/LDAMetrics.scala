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

package com.github.cloudml.zen.ml.clustering

import java.util.concurrent.ConcurrentLinkedQueue

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import breeze.numerics._
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.clustering.LDAPrecalc._
import com.github.cloudml.zen.ml.util.BVDecompressor
import com.github.cloudml.zen.ml.util.SimpleConcurrentBackend._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx2.impl.{ShippableVertexPartition => VertPartition}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.concurrent.Future


abstract class LDAMetrics(lda: LDA) extends Serializable {
  def getTotal: Double
  def calc(): LDAMetrics
  def calculated: Boolean
  def output(writer: String => Unit): Unit
}

abstract class LDACondMetrics(lda: LDA) extends LDAMetrics(lda) {
  def getWord: Double
  def getDoc: Double
}

class LDAPerplexity(lda: LDA) extends LDACondMetrics(lda) {
  var _calculated = false
  var pplx = 0.0
  var wpplx = 0.0
  var dpplx = 0.0

  override def getTotal: Double = pplx

  override def getWord: Double = wpplx

  override def getDoc: Double = dpplx

  override def calc(): LDAPerplexity = {
    val edges = lda.edges
    val verts = lda.verts
    val globalCountersBc = lda.globalCountersBc

    val algo = lda.algo
    val numTerms = lda.numTerms
    val numTokens = lda.numTokens
    val alpha = lda.alpha
    val beta = lda.beta
    val alphaAS = lda.alphaAS

    val newEdges = algo.refreshEdgeAssociations(edges, verts)
    val ppf = algo.perplexPartition(globalCountersBc, numTokens, numTerms, alpha, alphaAS, beta) _
    val sumPart = newEdges.partitionsRDD.mapPartitions(_.map { case (_, ep) =>
      ppf(ep)
    })
    val (llht, wllht, dllht) = sumPart.collect().unzip3

    pplx = math.exp(-llht.par.sum / numTokens)
    wpplx = math.exp(-wllht.par.sum / numTokens)
    dpplx = math.exp(-dllht.par.sum / numTokens)
    _calculated = true
    this
  }

  override def calculated: Boolean = _calculated

  override def output(writer: String => Unit): Unit = {
    if (calculated) {
      val o = s"total pplx=$getTotal, word pplx=$getWord, doc pplx=$getDoc"
      writer(o)
    } else {
      new Exception("Perplexity not calculated yet.")
    }
  }
}

class LDALogLikelihood(lda: LDA) extends LDACondMetrics(lda) {
  var _calculated = false
  var wllh = 0.0
  var dllh = 0.0

  override def getTotal: Double = wllh + dllh

  override def getWord: Double = wllh

  override def getDoc: Double = dllh

  override def calc(): LDALogLikelihood = {
    val edges = lda.edges
    val verts = lda.verts
    val globalCountersBc = lda.globalCountersBc

    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)
    val numTopics = lda.numTopics
    val numTerms = lda.numTerms
    val numDocs = lda.numDocs
    val numTokens = lda.numTokens
    val alpha = lda.alpha
    val beta = lda.beta
    val alphaAS = lda.alphaAS

    val alphaSum = alpha * numTopics
    val betaSum = beta * numTerms
    val lpf = calc_part(globalCountersBc, numTopics, numThreads, numTokens, alpha, beta, alphaAS) _
    val sumPart = verts.partitionsRDD.mapPartitions(_.map(lpf))
    val (wllht, dllht) = sumPart.collect().unzip

    val topicCounters = globalCountersBc.value
    val normWord = Range(0, numTopics).par.map(i => lgamma(topicCounters(i) + betaSum)).sum
    wllh = wllht.par.sum + numTopics * lgamma(betaSum) - normWord
    dllh = dllht.par.sum + numDocs * lgamma(alphaSum)
    _calculated = true
    this
  }

  private def calc_part(globalCountersBc: Broadcast[LDAGlobalCounters],
    numTopics: Int,
    numThreads: Int,
    numTokens: Long,
    alpha: Double,
    beta: Double,
    alphaAS: Double)
    (vp: VertPartition[TC]): (Double, Double) = {
    val topicCounters = globalCountersBc.value
    val alphaSum = alpha * numTopics
    val alphaRatio = calc_alphaRatio(numTopics, numTokens, alphaAS, alphaSum)
    val alphaks = calc_alphaks(topicCounters, alphaAS, alphaRatio)
    val lgamma_alphaks = alphaks.map(lgamma(_))
    val lgamma_beta = lgamma(beta)

    val totalSize = vp.capacity
    val index = vp.index
    val mask = vp.mask
    val values = vp.values
    @volatile var wllhs = 0.0
    @volatile var dllhs = 0.0
    val sizePerthrd = {
      val npt = totalSize / numThreads
      if (npt * numThreads == totalSize) npt else npt + 1
    }

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(Range(0, numThreads).iterator) { thid => withFuture {
      val decomp = new BVDecompressor(numTopics)
      val startPos = sizePerthrd * thid
      val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
      var wllh_th = 0.0
      var dllh_th = 0.0
      var pos = mask.nextSetBit(startPos)
      while (pos < endPos && pos >= 0) {
        val bv = decomp.CV2BV(values(pos))
        if (isTermId(index.getValue(pos))) {
          bv.activeValuesIterator.foreach(cnt =>
            wllh_th += lgamma(cnt + beta)
          )
          wllh_th -= bv.activeSize * lgamma_beta
        } else {
          val docTopics = bv.asInstanceOf[Ndk]
          val used = docTopics.used
          val index = docTopics.index
          val data = docTopics.data
          var nd = 0
          var i = 0
          while (i < used) {
            val topic = index(i)
            val cnt = data(i)
            dllh_th += lgamma(cnt + alphaks(topic)) - lgamma_alphaks(topic)
            nd += cnt
            i += 1
          }
          dllh_th -= lgamma(nd + alphaSum)
        }
        pos = mask.nextSetBit(pos + 1)
      }
      wllhs += wllh_th
      dllhs += dllh_th
    }}
    withAwaitReadyAndClose(all)

    (wllhs, dllhs)
  }

  override def calculated: Boolean = _calculated

  override def output(writer: String => Unit): Unit = {
    if (calculated) {
      val o = s"total llh=$getTotal, word llh=$getWord, doc llh=$getDoc"
      writer(o)
    } else {
      new Exception("Log-likelihood not calculated yet.")
    }
  }
}

class LDACoherence(lda: LDA,
  nTops: Int,
  lastIter: Boolean) extends LDAMetrics(lda) {
  var cohs: Array[Double] = _

  def getTotal: Double = if (cohs != null) cohs.sum / cohs.length else 0.0

  override def calc(): LDACoherence = if (lastIter) {
    val edges = lda.edges
    val verts = lda.verts

    val algo = lda.algo
    val numThreads = edges.context.getConf.getInt(cs_numThreads, 1)
    val numTopics = lda.numTopics
    val storageLevel = lda.storageLevel

    val topicTopWords = verts.partitionsRDD.mapPartitions(_.flatMap { vp =>
      val totalSize = vp.capacity
      val index = vp.index
      val mask = vp.mask
      val values = vp.values
      val topicWords = Array.fill(numTopics)(new ConcurrentLinkedQueue[(Int, Int)])

      val sizePerthrd = {
        val npt = totalSize / numThreads
        if (npt * numThreads == totalSize) npt else npt + 1
      }
      implicit val es = initExecutionContext(numThreads)
      val all = Future.traverse(Range(0, numThreads).iterator) { thid => withFuture {
        val decomp = new BVDecompressor(numTopics)
        val startPos = sizePerthrd * thid
        val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
        var pos = mask.nextSetBit(startPos)
        while (pos < endPos && pos >= 0) {
          val vid = index.getValue(pos)
          if (isTermId(vid)) {
            val wid = vid.toInt
            val termTopics = decomp.CV2BV(values(pos))
            termTopics match {
              case v: BDV[Count] =>
                var k = 0
                while (k < numTopics) {
                  val cnt = v(k)
                  if (cnt > 0) {
                    topicWords(k).add((cnt, wid))
                  }
                  k += 1
                }
              case v: BSV[Count] =>
                val used = v.used
                val index = v.index
                val data = v.data
                var i = 0
                while (i < used) {
                  topicWords(index(i)).add((data(i), wid))
                  i += 1
                }
            }
          }
          pos = mask.nextSetBit(pos + 1)
        }
      }}
      withAwaitReady(all)

      val all2 = Future.traverse(topicWords.iterator.zipWithIndex) { case (q, k) => withFuture {
        val topWords = q.toSeq.sorted.reverse.take(nTops).toArray
        (k, topWords)
      }}
      withAwaitResultAndClose(all2)
    }).reduceByKey((a, b) => (a ++ b).toSeq.sorted.reverse.take(nTops).toArray)
    topicTopWords.persist(storageLevel)

    val wordInParts = topicTopWords.mapPartitionsWithIndex { (pid, iter) =>
      val wordSet = iter.flatMap(_._2.map(_._2)).toSet
      wordSet.iterator.map((_, pid))
    }.aggregateByKey(new mutable.ArrayBuffer[Int])(_ += _, _ ++= _).mapValues(_.toArray)
    wordInParts.persist(storageLevel)

    val candWords = wordInParts.keys.collect().toSet
    val candWordsBc = edges.context.broadcast(candWords)

    val dwpf = algo.docOfWordsPartition(candWordsBc) _
    val docInWords = edges.partitionsRDD.mapPartitions(_.flatMap { case (_, ep) =>
      dwpf(ep)
    }).reduceByKey(_ ++ _).mapValues(_.sorted.toArray)

    val partitioner = topicTopWords.partitioner.get
    val numParts = partitioner.numPartitions
    val cohIndices = topicTopWords.mapPartitionsWithIndex( { case (pid, iter) =>
      val twp = iter.map { case (topic, topWords) => (topic, topWords.map(_._2)) }.toArray
      Iterator.single((pid, twp))
    }, preservesPartitioning=false).partitionBy(partitioner)
    val cohData = wordInParts.zipPartitions(docInWords, preservesPartitioning=false) { (partIter, docIter) =>
      val inParts = partIter.toMap
      val shippedData = Array.fill(numParts)(new ConcurrentLinkedQueue[(Int, Array[Long])])

      implicit val es = initExecutionContext(numThreads)
      val all = Future.traverse(docIter.grouped(numThreads * 5).toIterator) { batch => withFuture {
        batch.foreach { case (wid, docs) =>
          inParts(wid).foreach(shippedData(_).add((wid, docs)))
        }
      }}
      withAwaitReadyAndClose(all)
      shippedData.iterator.zipWithIndex.map { case (q, pid) => (pid, q.toSeq.toArray) }
    }.partitionBy(partitioner)

    val res = cohIndices.zipPartitions(cohData, preservesPartitioning=true) { (idxIter, dataIter) =>
      val inWords = new mutable.HashMap[Int, Array[Long]]
      dataIter.foreach(_._2.foreach(inWords += _))

      implicit val es = initExecutionContext(numThreads)
      val all = Future.traverse(idxIter.flatMap(_._2)) { case (topic, wids) => withFuture {
        val nWords = math.min(nTops, wids.length)
        var coh = 0.0
        var wi = 0
        while (wi < nWords - 1) {
          val datai = inWords(wids(wi))
          val ni = datai.length
          var wj = wi + 1
          while (wj < nWords) {
            val dataj = inWords(wids(wj))
            val nij = commonCount(datai, dataj)
            coh += math.log((nij + 1.0) / ni)
            wj += 1
          }
          wi += 1
        }
        (topic, coh)
      }}
      withAwaitResultAndClose(all)
    }.collect()

    topicTopWords.unpersist(blocking=false)
    wordInParts.unpersist(blocking=false)
    candWordsBc.unpersist(blocking=false)
    cohs = new Array[Double](numTopics)
    res.par.foreach { case (k, coh) => cohs(k) = coh }
    assert(cohs.forall(_ < 0))
    this
  } else {
    this
  }

  private def commonCount(docs1: Array[Long], docs2: Array[Long]): Int = {
    commonCount(docs1, 0, docs1.length, docs2, 0, docs2.length)
  }

  private def commonCount(docs1: Array[Long], s1: Int, e1: Int,
    docs2: Array[Long], s2: Int, e2: Int): Int = {
    if (s1 >= e1 || s2 >= e2) {
      0
    } else {
      val m1 = (s1 + e1) >> 1
      val m2 = (s2 + e2) >> 1
      val d1 = docs1(m1)
      val d2 = docs2(m2)
      if (d1 == d2) {
        commonCount(docs1, s1, m1, docs2, s2, m2) + commonCount(docs1, m1 + 1, e1, docs2, m2 + 1, e2) + 1
      } else if (d1 < d2) {
        commonCount(docs1, s1, m1 + 1, docs2, s2, m2) + commonCount(docs1, m1 + 1, e1, docs2, m2, e2)
      } else {
        commonCount(docs1, s1, m1, docs2, s2, m2 + 1) + commonCount(docs1, m1, e1, docs2, m2 + 1, e2)
      }
    }
  }

  override def calculated: Boolean = cohs != null

  override def output(writer: String => Unit): Unit = {
    if (calculated) {
      val o = s"average topic coherence=$getTotal"
      writer(o)
    } else {
      new Exception("Coherence not calculated yet.")
    }
  }
}

object LDAMetrics {
  def apply(lda: LDA,
    evalMetrics: Array[String],
    lastIter: Boolean): Array[LDAMetrics] = {
    evalMetrics.map { evalMetric =>
      val ldaMetric = evalMetric match {
        case "pplx" =>
          new LDAPerplexity(lda)
        case "llh" =>
          new LDALogLikelihood(lda)
        case "coh" =>
          new LDACoherence(lda, 20, lastIter)
      }
      ldaMetric.calc()
    }
  }
}
