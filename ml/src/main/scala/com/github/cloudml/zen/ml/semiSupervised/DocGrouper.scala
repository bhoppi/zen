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
import breeze.numerics._
import com.github.cloudml.zen.ml.sampler.CumulativeDist
import com.github.cloudml.zen.ml.util.XORShiftRandom


trait DocGrouper {
  def getGrp(docTopics: SparseVector[Int], docLen: Int): Int
}

abstract class BayesianGrouper(numGroups: Int,
  piGK: DenseMatrix[Float],
  priors: DenseVector[Double],
  burninIter: Int,
  sampIter: Int) extends DocGrouper {
  private val toSample = burninIter + 10 <= sampIter
  private lazy val gen = new XORShiftRandom()
  private lazy val samp = new CumulativeDist[Double]().reset(numGroups)

  def getGrp(docTopics: SparseVector[Int], docLen: Int): Int = {
    val llhs = priors.copy
    calcLlhs(docTopics, docLen.toDouble, llhs)
    if (toSample) {
      llhs :-= max(llhs)
      samp.resetDist(exp(llhs).iterator, numGroups).sampleRandom(gen)
    } else {
      argmax(llhs)
    }
  }

  def calcLlhs(docTopics: SparseVector[Int], nd: Double, llhs: DenseVector[Double]): Unit
}

class DirMultiGrouper(numGroups: Int,
  piGK: DenseMatrix[Float],
  priors: DenseVector[Double],
  burninIter: Int,
  sampIter: Int) extends BayesianGrouper(numGroups, piGK, priors, burninIter, sampIter) {
  override def calcLlhs(docTopics: SparseVector[Int], nd: Double, llhs: DenseVector[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val ndk = data(i)
      var g = 0
      while (g < numGroups) {
        val sgk = piGK(g, k) * nd
        llhs(g) += lgamma(sgk + ndk) - lgamma(sgk)
        g += 1
      }
      i += 1
    }
  }
}

class NaiveBayesGrouper(numGroups: Int,
  piGK: DenseMatrix[Float],
  priors: DenseVector[Double],
  burninIter: Int,
  sampIter: Int) extends BayesianGrouper(numGroups, piGK, priors, burninIter, sampIter) {
  override def calcLlhs(docTopics: SparseVector[Int], nd: Double, llhs: DenseVector[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val ndk = data(i)
      var g = 0
      while (g < numGroups) {
        llhs(g) += math.log(piGK(g, k)) * ndk
        g += 1
      }
      i += 1
    }
  }
}

abstract class MetricGrouper(numGroups: Int,
  eta: Double,
  piGK: DenseMatrix[Float]) extends DocGrouper {
  private val bases = calcBases()

  def calcBases(): DenseVector[Double] = DenseVector.zeros[Double](numGroups)

  def getGrp(docTopics: SparseVector[Int], docLen: Int): Int = {
    val metrics = bases.copy
    calcMetrics(docTopics, docLen.toDouble, metrics)
    argmin(metrics)
  }

  def calcMetrics(docTopics: SparseVector[Int], nd: Double, metrics: DenseVector[Double]): Unit
}

class KLDivergenceGrouper(numGroups: Int,
  eta: Double,
  piGK: DenseMatrix[Float]) extends MetricGrouper(numGroups, eta, piGK) {
  private val ln1pe = math.log(1.0 + eta)
  private val elned1pe = eta * (math.log(eta) - ln1pe)

  def calcMetrics(docTopics: SparseVector[Int], nd: Double, metrics: DenseVector[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val pk = data(i) / nd
      var g = 0
      while (g < numGroups) {
        val pigk = piGK(g, k)
        val rk = pk / pigk + eta
        metrics(g) += (rk * (math.log(rk) - ln1pe) - elned1pe) * pigk
        g += 1
      }
      i += 1
    }
  }
}

class BattacharyyaGrouper(numGroups: Int,
  eta: Double,
  piGK: DenseMatrix[Float]) extends MetricGrouper(numGroups, eta, piGK) {
  private val sqrt_eta = math.sqrt(eta)

  def calcMetrics(docTopics: SparseVector[Int], nd: Double, metrics: DenseVector[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val pk = data(i) / nd
      var g = 0
      while (g < numGroups) {
        val pigk = piGK(g, k)
        metrics(g) += (sqrt_eta - math.sqrt(pk / pigk + eta)) * pigk
        g += 1
      }
      i += 1
    }
  }
}

class EuclideanGrouper(numGroups: Int,
  eta: Double,
  piGK: DenseMatrix[Float]) extends MetricGrouper(numGroups, eta, piGK) {
  override def calcBases(): DenseVector[Double] = {
    convert(sum(piGK :* piGK, Axis._1).slice(0, numGroups), Double)
  }

  def calcMetrics(docTopics: SparseVector[Int], nd: Double, metrics: DenseVector[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val pk = data(i) / nd
      var g = 0
      while (g < numGroups) {
        metrics(g) += pk * (pk - 2.0 * piGK(g, k))
        g += 1
      }
      i += 1
    }
  }
}

class GroupContext(numGroups: Int,
  eta: Double,
  priors: DenseVector[Double],
  burninIter: Int,
  docGrouperStr: String) extends Serializable {
  private var sampIter = 0

  def setIteration(sampIter: Int): Unit = {
    this.sampIter = sampIter
  }

  def isBurnin: Boolean = sampIter <= burninIter

  def totalGroups: Int = if (isBurnin) numGroups + 1 else numGroups

  def getDocGrouper(piGK: DenseMatrix[Float]): DocGrouper = {
    docGrouperStr match {
      case "dirmulti" =>
        new DirMultiGrouper(numGroups, piGK, priors, burninIter, sampIter)
      case "naivebayes" =>
        new NaiveBayesGrouper(numGroups, piGK, priors, burninIter, sampIter)
      case "kldivergence" =>
        new KLDivergenceGrouper(numGroups, eta, piGK)
      case "battacharyya" =>
        new BattacharyyaGrouper(numGroups, eta, piGK)
      case "euclidean" =>
        new EuclideanGrouper(numGroups, eta, piGK)
      case _ =>
        throw new NoSuchMethodException("Not implemented.")
    }
  }
}
