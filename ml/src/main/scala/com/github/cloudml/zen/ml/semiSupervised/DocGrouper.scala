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


trait DocGrouper extends Serializable

class DirMultiGrouper(numGroups: Int,
  piGK: DenseMatrix[Float],
  priors: DenseVector[Double],
  burninIter: Int,
  sampIter: Int) extends DocGrouper {
  private val gen = new XORShiftRandom()
  private val samp = new CumulativeDist[Double]().reset(numGroups)

  def getGrp(docTopics: SparseVector[Int], docLen: Int): Int = {
    val llhs = priors.copy
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
    if (sampIter <= burninIter) {
      llhs :-= max(llhs)
      samp.resetDist(exp(llhs).iterator, numGroups).sampleRandom(gen)
    } else {
      argmax(llhs)
    }
  }
}
