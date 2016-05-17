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

import spire.math.{Numeric => spNum}


trait DiscreteSampler[@specialized(Double, Int, Float, Long) T] extends Sampler[T] {
  def length: Int
  def used: Int
  def update(state: Int, value: => T): Unit
  def deltaUpdate(state: Int, delta: => T): Unit
  def resetDist(probs: Array[T], space: Array[Int], psize: Int): DiscreteSampler[T]
  def resetDist(distIter: Iterator[(Int, T)], psize: Int): DiscreteSampler[T]
  def reset(newSize: Int): DiscreteSampler[T]

  @tailrec final def resampleRandom(gen: Random,
    state: Int,
    residualRate: => Double,
    numResampling: Int = 2)(implicit ev: spNum[T]): Int = {
    val newState = sampleRandom(gen)
    if (newState == state && numResampling >= 0 && used > 1) {
      val r = residualRate
      if (residualRate >= 1.0 || gen.nextDouble() < residualRate) {
        resampleRandom(gen, state, r, numResampling - 1)
      } else {
        newState
      }
    } else {
      newState
    }
  }

  @tailrec final def resampleFrom(base: T,
    gen: Random,
    state: Int,
    residualRate: => Double,
    numResampling: Int = 2)(implicit ev: spNum[T]): Int = {
    val newState = sampleFrom(base, gen)
    if (newState == state && numResampling >= 0 && used > 1) {
      val r = residualRate
      if (r >= 1.0 || gen.nextDouble() < r) {
        val newBase = ev.fromDouble(gen.nextDouble() * ev.toDouble(norm))
        resampleFrom(newBase, gen, state, r, numResampling - 1)
      } else {
        newState
      }
    } else {
      newState
    }
  }
}
