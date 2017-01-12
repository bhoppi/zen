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

import breeze.linalg._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.Concurrent._
import com.github.cloudml.zen.ml.util.{BVDecompressor, CompressedVector}

import scala.collection.concurrent.TrieMap


abstract class GLDAMetrics(glda: GLDA) extends Serializable {
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
    val numTerms = glda.numTerms
    val params = glda.params
    val globalVarsBc = glda.globalVarsBc

    val shippeds = glda.algo.ShipParaBlocks(dataBlocks, paraBlocks)
    // Below identical map is used to isolate the impact of locality of CheckpointRDD
    val isoRDD = dataBlocks.mapPartitions(_.seq, preservesPartitioning=true)
    val totalLlh = isoRDD.zipPartitions(shippeds, preservesPartitioning=true) { (dataIter, shpsIter) =>
      dataIter.map { case (_, DataBlock(termRecs, docRecs)) =>
        val totalDocSize = docRecs.length
        implicit val es = newExecutionContext(numThreads)

        // Stage 1: assign all termTopics
        val termVecs = new TrieMap[Int, CompressedVector]()
        parallelized_foreachElement[(Int, ShippedAttrsBlock)](shpsIter, numThreads, (shp, _) => {
          val (_, ShippedAttrsBlock(termIds, termAttrs)) = shp
          termIds.iterator.zip(termAttrs.iterator).foreach { case (termId, termAttr) =>
            termVecs(termId) = termAttr
          }
        })

        // Stage 2: calculate all docTopics, docLen, docGrp
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

        // Stage 3: Calc perplexity
        val eta = params.eta
        val mu = params.mu
        val decomps = Array.fill(numThreads)(new BVDecompressor(numTopics))
        val GlobalVars(piGK, nK, _) = globalVarsBc.value
        val megSums = calcSum_megDenses(piGK, eta, mu)
        parallelized_reduceBatch[TermRec, Double](termRecs.iterator, numThreads, 100, (batch, _, thid) => {
          val decomp = decomps(thid)
          var llhSum = 0.0
          batch.foreach { termRec =>
            val TermRec(termId, termData) = termRec
            val termTopics = decomp.CV2BV(termVecs(termId))
            val denseTermTopics = getDensed(termTopics)
            val tegSums = calcSum_tegSparses(piGK, nK, termTopics, eta)
            val calcSum_dtmSparse_f = calcSum_dtmSparse(nK, denseTermTopics) _
            var i = 0
            while (i < termData.length) {
              val docPos = termData(i)
              val docI = termData(i + 1)
              val docData = docRecs(docPos).docData
              val (docTopics, docLen, g) = docResults(docPos)
              var totalSum = megSums(g) + tegSums(g) + calcSum_dtmSparse_f(docTopics, docLen, mu)
              totalSum /= (1f + eta) * (1f + mu * numTerms)
              val ind = docData(docI)
              val termCnt = if (ind >= 0) 1 else -ind
              llhSum += termCnt * math.log(totalSum)
              i += 2
            }
          }
          llhSum
        }, _ + _, closing=true)
      }
    }.reduce(_ + _)

    pplx = math.exp(-totalLlh / glda.numTokens)
    _calculated = true
    this
  }

  def calcSum_megDenses(piGK: DenseMatrix[Float], eta: Float, mu: Float): DenseVector[Float] = {
    sum(piGK :* (eta * mu), Axis._1)
  }

  def calcSum_tegSparses(piGK: DenseMatrix[Float],
    nK: Array[Int],
    termTopics: Vector[Int],
    eta: Float): DenseVector[Float] = {
    val tegSums = DenseVector.zeros[Float](piGK.rows)
    termTopics match {
      case v: DenseVector[Int] =>
        var k = 0
        while (k < piGK.cols) {
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

  def calcSum_dtmSparse(nK: Array[Int],
    denseTermTopics: DenseVector[Int])(docTopics: SparseVector[Int],
    docLen: Int,
    mu: Float): Float = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var sum = 0f
    var i = 0
    while (i < used) {
      val k = index(i)
      sum += (mu + denseTermTopics(k).toFloat / nK(k)) * data(i) / docLen
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
  def apply(glda: GLDA,
    evalMetrics: Array[String]): Array[GLDAMetrics] = {
    evalMetrics.map { evalMetric =>
      val ldaMetric = evalMetric match {
        case "pplx" =>
          new GLDAPerplexity(glda)
      }
      ldaMetric.calc()
    }
  }
}
