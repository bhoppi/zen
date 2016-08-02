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

import java.util.concurrent.atomic.AtomicIntegerArray

import breeze.linalg.{sum, DenseVector => BDV, SparseVector => BSV}
import com.github.cloudml.zen.ml.clustering.LDADefines._
import com.github.cloudml.zen.ml.util.BVCompressor
import com.github.cloudml.zen.ml.util.Concurrent._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx2.impl.{EdgePartition, ShippableVertexPartition => VertPartition}

import scala.collection.mutable
import scala.concurrent.Future


abstract class LDATrainer(numTopics: Int, numThreads: Int)
  extends LDAAlgorithm(numTopics, numThreads) {
  def isByDoc: Boolean

  def perplexPartition(globalCountersBc: Broadcast[LDAGlobalCounters],
    numTokens: Long,
    numTerms: Int,
    alpha: Double,
    alphaAS: Double,
    beta: Double)
    (ep: EdgePartition[TA, Nvk]): (Double, Double, Double)

  def docOfWordsPartition(candWordsBc: Broadcast[Set[Int]])
    (ep: EdgePartition[TA, _]): Iterator[(Int, mutable.ArrayBuffer[Long])]

  def aggregateCounters(vp: VertPartition[TC], cntsIter: Iterator[NvkPair]): VertPartition[TC] = {
    val totalSize = vp.capacity
    val index = vp.index
    val mask = vp.mask
    val values = vp.values
    val results = new Array[Nvk](totalSize)
    val marks = new AtomicIntegerArray(totalSize)

    implicit val es = initExecutionContext(numThreads)
    val all = Future.traverse(cntsIter.grouped(numThreads * 5).toIterator) { batch => withFuture {
      batch.foreach { case (vid, counter) =>
        val i = index.getPos(vid)
        if (marks.getAndDecrement(i) == 0) {
          results(i) = counter
        } else {
          while (marks.getAndSet(i, -1) <= 0) {}
          val agg = results(i)
          results(i) = if (isTermId(vid)) agg match {
            case u: BDV[Count] => counter match {
              case v: BDV[Count] => u :+= v
              case v: BSV[Count] => u :+= v
            }
            case u: BSV[Count] => counter match {
              case v: BDV[Count] => v :+= u
              case v: BSV[Count] =>
                u :+= v
                if (u.activeSize >= dscp) toBDV(u) else u
            }
          } else {
            agg.asInstanceOf[Ndk] :+= counter.asInstanceOf[Ndk]
          }
        }
        marks.set(i, Int.MaxValue)
      }
    }}
    withAwaitReady(all)

    // compress counters
    val sizePerthrd = {
      val npt = totalSize / numThreads
      if (npt * numThreads == totalSize) npt else npt + 1
    }
    val all2 = Future.traverse(Range(0, numThreads).iterator) { thid => withFuture {
      val comp = new BVCompressor(numTopics)
      val startPos = sizePerthrd * thid
      val endPos = math.min(sizePerthrd * (thid + 1), totalSize)
      var pos = mask.nextSetBit(startPos)
      while (pos < endPos && pos >= 0) {
        values(pos) = comp.BV2CV(results(pos))
        pos = mask.nextSetBit(pos + 1)
      }
    }}
    withAwaitReadyAndClose(all2)

    vp.withValues(values)
  }

  def sum_abDense(alphak_denoms: BDV[Double],
    beta: Double): Double = {
    sum(alphak_denoms.copy :*= beta)
  }
}

object LDATrainer {
  def initAlgorithm(algoStr: String, numTopics: Int, numThreads: Int): LDATrainer = {
    algoStr.toLowerCase match {
      case "zenlda" =>
        println("using ZenLDA sampling algorithm.")
        new ZenLDA(numTopics, numThreads)
      case "lightlda" =>
        println("using LightLDA sampling algorithm.")
        new LightLDA(numTopics, numThreads)
      case "f+lda" =>
        println("using F+LDA sampling algorithm.")
        new FPlusLDA(numTopics, numThreads)
      case "aliaslda" =>
        println("using AliasLDA sampling algorithm.")
        new AliasLDA(numTopics, numThreads)
      case "sparselda" =>
        println("using SparseLDA sampling algorithm")
        new SparseLDA(numTopics, numThreads)
      case _ =>
        throw new NoSuchMethodException("No this algorithm or not implemented.")
    }
  }
}
