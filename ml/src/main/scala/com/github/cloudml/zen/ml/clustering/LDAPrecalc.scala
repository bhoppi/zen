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

import breeze.linalg.{convert, DenseVector => BDV}
import com.github.cloudml.zen.ml.clustering.LDADefines._


object LDAPrecalc {
  @inline def calc_alphaRatio(numTopics: Int,
    numTokens: Long,
    alphaAS: Double,
    alphaSum: Double): Double = {
    alphaSum / (numTokens + alphaAS * numTopics)
  }

  def calc_denoms(topicCounters: BDV[Count],
    numTopics: Int,
    betaSum: Double): BDV[Double] = {
    val arr = new Array[Double](numTopics)
    var i = 0
    while (i < numTopics) {
      arr(i) = 1.0 / (topicCounters(i) + betaSum)
      i += 1
    }
    new BDV(arr)
  }

  def calc_alphak_denoms(denoms: BDV[Double],
    alphaAS: Double,
    betaSum: Double,
    alphaRatio: Double): BDV[Double] = {
    (denoms.copy :*= ((alphaAS - betaSum) * alphaRatio)) :+= alphaRatio
  }

  def calc_beta_denoms(denoms: BDV[Double],
    beta: Double): BDV[Double] = {
    denoms.copy :*= beta
  }

  def calc_alphaks(topicCounters: BDV[Count],
    alphaAS: Double,
    alphaRatio: Double): BDV[Double] = {
    (convert(topicCounters, Double) :+= alphaAS) :*= alphaRatio
  }
}
