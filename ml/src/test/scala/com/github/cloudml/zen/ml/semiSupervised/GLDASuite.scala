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

import java.io.File
import java.util.Random

import breeze.linalg._
import breeze.stats.distributions.Poisson
import com.github.cloudml.zen.ml.sampler.AliasTable
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.SharedSparkContext
import com.google.common.io.Files
import org.apache.spark.storage.StorageLevel
import org.scalatest.FunSuite


class GLDASuite extends FunSuite with SharedSparkContext {
  import GLDASuite._

  test("GLDA || Gibbs sampling") {
    val model = genGLDAModel()
    val corpus = sampleCorpus(model)
    val data = sc.parallelize(corpus, 2)

    val dataBlocks = GLDA.convertBowDocs(data, numTopics, numThreads)
    val paraBlocks = GLDA.buildParaBlocks(dataBlocks)
    val params = HyperParams(alpha, beta, eta, mu)
    val glda = GLDA((dataBlocks, paraBlocks), numTopics, numGroups, numThreads, params, storageLevel)

    val pps = new Array[Double](increIter)
    val startedAt = System.nanoTime
    for (i <- 1 to increIter) {
      glda.fit(totalIter)
      pps(i - 1) = GLDAMetrics(glda, evalMetrics)(0).getTotal
    }
    println((System.nanoTime - startedAt) / 1e9)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ > 0).toDouble / ppsDiff.length > 0.6)
    assert(pps.head - pps.last > 0)

    val gldaModel = glda.toGLDAModel
    val tempDir = Files.createTempDir()
    tempDir.deleteOnExit()
    val path = tempDir.toURI.toString + File.separator + "glda"
    gldaModel.save(sc, path)
    val sameModel = GLDAModel.load(sc, path)
    assert(sameModel.params === gldaModel.params)
  }
}

object GLDASuite {
  val numTopics = 50
  val numGroups = 2
  val numTerms = 1000
  val numDocs = 100
  val expDocLen = 300
  val alpha = 1f
  val beta = 1f
  val eta = 0.1f
  val mu = 0.1f
  val totalIter = 2
  val burninIter = 1
  val increIter = 10
  val numThreads = 2
  val storageLevel = StorageLevel.MEMORY_AND_DISK
  val evalMetrics = Array("pplx")

  def genGLDAModel(): Array[Array[Double]] = {
    val model = Array.ofDim[Double](numTopics, numTerms)
    val width = numTerms.toDouble / numTopics
    for (topic <- 0 until numTopics) {
      val topicTermDist = model(topic)
      val topicCentroid = width * (topic + 1)
      for (i <- 0 until numTerms) {
        // treat the term list as a circle, so the distance between the first one and the last one
        // is 1, not n-1.
        val distance = Math.abs(topicCentroid - i) % (numTerms / 2)
        // Possibility is decay along with distance
        topicTermDist(i) = 1.0 / (1.0 + distance)
      }
    }
    model
  }

  def sampleCorpus(model: Array[Array[Double]]): Array[DocBow] = {
    val gen = new Random()
    val topicTermSamps = model.map(dist =>
      new AliasTable[Double]().resetDist(dist, null, numTerms)
    )
    val docTopicSamp = new AliasTable[Int]()
    Array.tabulate(numDocs) { i =>
      val docLen = new Poisson(expDocLen).sample()
      val docK = gen.nextInt(numTopics >> 2) + 1
      val docTopicDist = new Array[Int](numTopics)
      for (_ <- 0 until docK) {
        docTopicDist(gen.nextInt(numTopics)) += 1
      }
      docTopicSamp.resetDist(docTopicDist, null, numTopics)
      val terms = mixtureSampler(gen, topicTermSamps, docTopicSamp, docLen)
      val bow = SparseVector.zeros[Int](numTerms)
      terms.foreach { term => bow(term) += 1 }
      DocBow(i.toLong, gen.nextInt(numGroups), bow)
    }
  }

  def mixtureSampler(gen: Random,
    topicTermSamps: Array[AliasTable[Double]],
    docTopicSamp: AliasTable[Int],
    docLen: Int): Array[Int] = {
    Array.fill(docLen)(topicTermSamps(docTopicSamp.sampleRandom(gen)).sampleRandom(gen))
  }
}
