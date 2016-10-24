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

import breeze.linalg._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.{BVDecompressor, CompressedVector}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.concurrent.Future


abstract class GLDAMetrics(lda: GLDA) extends Serializable {
  def getTotal: Double
  def calc(): GLDAMetrics
  def calculated: Boolean
  def output(writer: String => Unit): Unit
}

class GLDAPerplexity(glda: GLDA) extends GLDAMetrics(glda) {
  var _calculated = false
  var pplx = 0.0

  override def getTotal: Double = pplx

  override def calc(): GLDAPerplexity = {
    val dataBlocks = glda.dataBlocks
    val paraBlocks = glda.paraBlocks
    val numTopics = glda.numTopics
    val numThreads = glda.numThreads
    val params = glda.params
    val globalVarsBc = glda.globalVarsBc

    val shippeds = glda.algo.ShipParaBlocks(dataBlocks, paraBlocks)
    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = dataBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    val totalLlh = isoRDD.zipPartitions(shippeds, preservesPartitioning=true) { (dataIter, shpsIter) =>
      val GlobalVars(piGK, sigGW, nK, _) = globalVarsBc.value
      dataIter.map { case (pid, DataBlock(termRecs, docRecs)) =>
        // Stage 1: assign all termTopics
        val termVecs = new mutable.HashMap[Int, CompressedVector]()
        implicit val es = initExecutionContext(numThreads)
        val allAssign = shpsIter.map(shp => withFuture {
          val (_, ShippedAttrsBlock(termIds, termAttrs)) = shp
          termIds.iterator.zip(termAttrs.iterator).foreach { case (termId, termAttr) =>
            termVecs(termId) = termAttr
          }
        })
        withAwaitReady(Future.sequence(allAssign))

        // Stage 2: calculate all docTopics, docLen, docGrp
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

        // Stage 3: Calc perplexity
        val eta = params.eta
        val mu = params.mu
        val thq = new ConcurrentLinkedQueue(1 to numThreads)
        val decomps = Array.fill(numThreads)(new BVDecompressor(numTopics))
        val egSums = calcSum_egDenses(piGK, eta)
        val allPplx = termRecs.iterator.map(termRec => withFuture {
          val thid = thq.poll() - 1
          var llhSum = 0.0
          try {
            val decomp = decomps(thid)

            val TermRec(termId, termData) = termRec
            val termTopics = decomp.CV2BV(termVecs(termId))
            val term_sigG = sigGW(::, termId)
            val denseTermTopics = getDensed(termTopics)
            val tegSums = calcSum_tegSparses(piGK, nK, termTopics, eta)
            val calcSum_dtmSparse_f = calcSum_dtmSparse(nK, denseTermTopics) _
            var i = 0
            while (i < termData.length) {
              val docPos = termData(i)
              val docI = termData(i + 1)
              val docData = docRecs(docPos).docData
              val (docTopics, docLen, g) = docResults(docPos)
              val muSig = mu * term_sigG(g)
              var totalSum = muSig * egSums(g) + tegSums(g)
              val ind = docData(docI)
              totalSum += calcSum_dtmSparse_f(docTopics, docLen, muSig)
              val termCnt = if (ind >= 0) 1 else -ind
              llhSum += termCnt * math.log(totalSum)
              i += 2
            }
          } finally {
            thq.add(thid + 1)
          }
          llhSum
        })
        withAwaitResultAndClose(Future.reduce(allPplx)(_ + _))
      }
    }.reduce(_ + _)

    pplx = math.exp(-totalLlh / glda.numTokens)
    this
  }

  def calcSum_egDenses(piGK: DenseMatrix[Float], eta: Float): DenseVector[Float] = {
    sum(piGK :* eta, Axis._1)
  }

  def calcSum_tegSparses(piGK: DenseMatrix[Float],
    nK: DenseVector[Int],
    termTopics: Vector[Int],
    eta: Float): DenseVector[Float] = {
    val numTopics = piGK.cols
    val numGroups = piGK.rows
    val tegSums = DenseVector.zeros[Float](numGroups)
    termTopics match {
      case v: DenseVector[Int] =>
        var k = 0
        while (k < numTopics) {
          val cnt = v(k)
          if (cnt > 0) {
            tegSums :+= piGK(::, k) :* (eta * cnt / nK(k))
          }
          k += 1
        }
      case v: SparseVector[Int] =>
        val used = v.used
        val index = v.index
        val data = v.data
        var i = 0
        while (i < used) {
          val k = index(i)
          tegSums :+= piGK(::, k) :* (eta * data(i) / nK(k))
          i += 1
        }
    }
    tegSums
  }

  def calcSum_dtmSparse(nK: DenseVector[Int],
    denseTermTopics: DenseVector[Int])(docTopics: SparseVector[Int],
    docLen: Int,
    muSig: Float): Float = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var sum = 0f
    var i = 0
    while (i < used) {
      val k = index(i)
      sum += (muSig + denseTermTopics(k).toFloat / nK(k)) * data(i) / docLen
      i += 1
    }
    sum
  }

  override def calculated: Boolean = _calculated

  override def output(writer: String => Unit): Unit = {
    if (calculated) {
      val o = s"perplexity=$getTotal"
      writer(o)
    } else {
      new Exception("Perplexity not calculated yet.")
    }
  }
}

object GLDAMetrics {
  def apply(lda: GLDA,
    evalMetrics: Array[String]): Array[GLDAMetrics] = {
    evalMetrics.map { evalMetric =>
      val ldaMetric = evalMetric match {
        case "pplx" =>
          new GLDAPerplexity(lda)
      }
      ldaMetric.calc()
    }
  }
}
