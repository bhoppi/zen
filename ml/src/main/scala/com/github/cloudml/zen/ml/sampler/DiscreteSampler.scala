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

package com.github.cloudml.zen.ml.sampler

import java.util.Random

import scala.annotation.tailrec


trait DiscreteSampler[@specialized(Double, Int, Float, Long) T] extends Sampler[T] {
  private var residualRate: (Int) => Double = _

  def length: Int
  def used: Int
  def update(state: Int, value: => T): Unit
  def deltaUpdate(state: Int, delta: => T): Unit
  def resetDist(probs: Array[T], space: Array[Int], psize: Int): DiscreteSampler[T]
  def resetDist(distIter: Iterator[(Int, T)], psize: Int): DiscreteSampler[T]
  def reset(newSize: Int): DiscreteSampler[T]

  @inline def setResidualRate(rrf: (Int) => Double): Unit = {
    residualRate = rrf
  }

  @inline def unsetResidualRate(): Unit = {
    residualRate = null
  }

  def resampleFrom(base: T,
    gen: Random,
    state: Int): Int = {
    val newState = sampleFrom(base, gen)
    if (residualRate != null && newState == state && used > 1) {
      val r = residualRate(state)
      if (r >= 1.0 || gen.nextDouble() < r) {
        doResampling(gen, state, r)
      } else {
        newState
      }
    } else {
      newState
    }
  }

  @tailrec private final def doResampling(gen: Random,
    state: Int,
    rate: Double,
    numResampling: Int = 2): Int = {
    if (numResampling > 0) {
      val newState = sampleRandom(gen)
      if (newState == state && gen.nextDouble() < rate) {
        doResampling(gen, state, rate, numResampling - 1)
      } else {
        newState
      }
    } else {
      state
    }
  }
}
