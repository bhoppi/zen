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


trait DocGrouper extends Serializable {
  def getGrp(docTopics: SparseVector[Int], docLen: Int): Int
}

abstract class BayesianGrouper(numGroups: Int,
  piGK: DenseMatrix[Float],
  priors: DenseVector[Double],
  burninIter: Int,
  sampIter: Int) extends DocGrouper {
  protected val gen = new XORShiftRandom()
  protected val samp = new CumulativeDist[Double]().reset(numGroups)

  def getGrp(docTopics: SparseVector[Int], docLen: Int): Int = {
    val llhs = priors.copy
    calcLlhs(docTopics, docLen, llhs)
    if (sampIter <= burninIter) {
      llhs :-= max(llhs)
      samp.resetDist(exp(llhs).iterator, numGroups).sampleRandom(gen)
    } else {
      argmax(llhs)
    }
  }

  def calcLlhs(docTopics: SparseVector[Int], docLen: Int, llhs: DenseVector[Double]): Unit
}

class DirMultiGrouper(numGroups: Int,
  piGK: DenseMatrix[Float],
  priors: DenseVector[Double],
  burninIter: Int,
  sampIter: Int) extends BayesianGrouper(numGroups, piGK, priors, burninIter, sampIter) {
  override def calcLlhs(docTopics: SparseVector[Int], docLen: Int, llhs: DenseVector[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val ndk = data(i)
      var g = 0
      while (g < numGroups) {
        val sgk = (piGK(g, k) * docLen).toDouble
        llhs(g) += lgamma(sgk + ndk) - lgamma(sgk)
        g += 1
      }
      i += 1
    }
  }
}

class DiscreteGrouper(numGroups: Int,
  piGK: DenseMatrix[Float],
  priors: DenseVector[Double],
  burninIter: Int,
  sampIter: Int) extends BayesianGrouper(numGroups, piGK, priors, burninIter, sampIter) {
  override def calcLlhs(docTopics: SparseVector[Int], docLen: Int, llhs: DenseVector[Double]): Unit = {
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
  piGK: DenseMatrix[Float]) extends DocGrouper {
  def getGrp(docTopics: SparseVector[Int], docLen: Int): Int = {
    val metrics = new Array[Double](numGroups)
    calcMetrics(docTopics, docLen, metrics)
    argmax(metrics)
  }

  def calcMetrics(docTopics: SparseVector[Int], docLen: Int, metrics: Array[Double]): Unit
}

class BtchyGrouper(numGroups: Int,
  piGK: DenseMatrix[Float]) extends MetricGrouper(numGroups, piGK) {
  def calcMetrics(docTopics: SparseVector[Int], docLen: Int, metrics: Array[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val pk = data(i) / docLen.toDouble
      var g = 0
      while (g < numGroups) {
        metrics(g) += math.sqrt(piGK(g, k) * pk)
        g += 1
      }
      i += 1
    }
  }
}

class EuclideanGrouper(numGroups: Int,
  piGK: DenseMatrix[Float]) extends MetricGrouper(numGroups, piGK) {
  def calcMetrics(docTopics: SparseVector[Int], docLen: Int, metrics: Array[Double]): Unit = {
    val used = docTopics.used
    val index = docTopics.index
    val data = docTopics.data
    var i = 0
    while (i < used) {
      val k = index(i)
      val pk = data(i) / docLen.toDouble
      var g = 0
      while (g < numGroups) {
        val delta = piGK(g, k) - pk
        metrics(g) += delta * delta
        g += 1
      }
      i += 1
    }
  }
}

object DocGrouper {
  def apply(dgStr: String,
    numGroups: Int,
    piGK: DenseMatrix[Float],
    priors: DenseVector[Double],
    burninIter: Int,
    sampIter: Int): DocGrouper = {
    dgStr.toLowerCase match {
      case "dirmulti" =>
        new DirMultiGrouper(numGroups, piGK, priors, burninIter, sampIter)
      case "discrete" =>
        new DiscreteGrouper(numGroups, piGK, priors, burninIter, sampIter)
      case "battacharyya" =>
        new BtchyGrouper(numGroups, piGK)
      case "euclidean" =>
        new EuclideanGrouper(numGroups, piGK)
      case _ =>
        throw new NoSuchMethodException("Not implemented.")
    }
  }
}
