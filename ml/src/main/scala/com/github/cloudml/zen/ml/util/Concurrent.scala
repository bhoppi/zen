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

package com.github.cloudml.zen.ml.util

import java.util.concurrent.{ConcurrentLinkedQueue, Executors, LinkedBlockingQueue, ThreadPoolExecutor}

import scala.collection.JavaConversions._
import scala.concurrent._
import scala.concurrent.duration._


object Concurrent extends Serializable {
  type ThID = Int

  @inline def withFuture[T](body: => T)(implicit es: ExecutionContextExecutorService): Future[T] = {
    Future(body)(es)
  }

  @inline def withAwaitReady[T](future: Future[T]): Unit = {
    Await.ready(future, 1.hour)
  }

  def withAwaitReadyAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): Unit = {
    Await.ready(future, 1.hour)
    closeExecutionContext(es)
  }

  @inline def withAwaitResult[T](future: Future[T]): T = {
    Await.result(future, 1.hour)
  }

  def withAwaitResultAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): T = {
    val res = Await.result(future, 1.hour)
    closeExecutionContext(es)
    res
  }

  @inline def initExecutionContext(numThreads: Int): ExecutionContextExecutorService = {
    ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(numThreads))
  }

  @inline def closeExecutionContext(es: ExecutionContextExecutorService): Unit = {
    es.shutdown()
  }

  // coarse-grained multi-threading, simple
  def parallelized_foreachSplit(totalSize: Int,
    nThreads: Int,
    funcThrd: (Int, Int, ThID) => Unit,
    closing: Boolean = false)(implicit es: ExecutionContextExecutorService): Unit = {
    val sizePerThrd = {
      val npt = totalSize / nThreads
      if (npt * nThreads == totalSize) npt else npt + 1
    }
    val all = Range(0, nThreads).map(thid => withFuture {
      val is = sizePerThrd * thid
      val in = math.min(is + sizePerThrd, totalSize)
      funcThrd(is, in, thid)
    })
    val bf = if (closing) withAwaitReadyAndClose _ else withAwaitReady _
    bf(Future.sequence(all))
  }

  // work-stealing mode multi-threading, more load-balanced
  def parallelized_foreachElement[T](totalIter: Iterator[T],
    nThreads: Int,
    funcThrd: (T, ThID) => Unit,
    closing: Boolean = false)(implicit es: ExecutionContextExecutorService): Unit = {
    val thq = new ConcurrentLinkedQueue(1 to nThreads)
    val all = totalIter.map(e => withFuture {
      val thid = thq.poll() - 1
      try {
        funcThrd(e, thid)
      } finally {
        thq.add(thid + 1)
      }
    })
    val bf = if (closing) withAwaitReadyAndClose _ else withAwaitReady _
    bf(Future.sequence(all))
  }

  // mini-batch between coarse-grained & work-stealing, most efficient
  def parallelized_foreachBatch[T](totalIter: Iterator[T],
    nThreads: Int,
    nBatch: Int,
    funcThrd: (Seq[T], Int, ThID) => Unit,
    closing: Boolean = false)(implicit es: ExecutionContextExecutorService): Unit = {
    val thq = new ConcurrentLinkedQueue(1 to nThreads)
    val all = totalIter.grouped(nBatch).zipWithIndex.map { case (batch, bi) =>
      withFuture {
        val thid = thq.poll() - 1
        try {
          funcThrd(batch, bi, thid)
        } finally {
          thq.add(thid + 1)
        }
      }
    }
    val bf = if (closing) withAwaitReadyAndClose _ else withAwaitReady _
    bf(Future.sequence(all))
  }

  def parallelized_mapElement[T, U](totalIter: Iterator[T],
    nThreads: Int,
    funcThrd: T => U,
    closing: Boolean = false)(implicit es: ExecutionContextExecutorService): Iterator[U] = {
    val all = totalIter.map(e => withFuture(funcThrd(e)))
    val bf = if (closing) withAwaitResultAndClose _ else withAwaitResult _
    bf(Future.sequence(all))
  }

  def parallelized_mapBatch[T, U](totalIter: Iterator[T],
    nThreads: Int,
    funcThrd: T => U,
    closing: Boolean = false)(implicit es: ExecutionContextExecutorService): Iterator[U] = {
    totalIter.grouped(nThreads).flatMap { batch =>
      val all = Future.traverse(batch)(e => withFuture(funcThrd(e)))
      withAwaitResult(all)
    } ++ {
      if (closing) {
        closeExecutionContext(es)
      }
      Iterator.empty
    }
  }

  def parallelized_reduceSplit[U](totalSize: Int,
    nThreads: Int,
    funcThrd: (Int, Int) => U,
    reducer: (U, U) => U,
    closing: Boolean = false)(implicit es: ExecutionContextExecutorService): U = {
    val sizePerThrd = {
      val npt = totalSize / nThreads
      if (npt * nThreads == totalSize) npt else npt + 1
    }
    val all = Range(0, nThreads).map(thid => withFuture {
      val is = sizePerThrd * thid
      val in = math.min(is + sizePerThrd, totalSize)
      funcThrd(is, in)
    })
    val bf = if (closing) withAwaitResultAndClose _ else withAwaitResult _
    bf(Future.reduce(all)(reducer))
  }
}

object DebugConcurrent extends Serializable {
  def withFuture[T](body: => T)(implicit es: ExecutionContextExecutorService): Future[T] = {
    val future = Future(body)(es)
    future.onFailure { case e =>
      e.printStackTrace()
    }(scala.concurrent.ExecutionContext.Implicits.global)
    future
  }

  def withAwaitReady[T](future: Future[T]): Unit = {
    Await.ready(future, 1.hour)
  }

  def withAwaitReadyAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): Unit = {
    future.onComplete { _ =>
      closeExecutionContext(es)
    }(scala.concurrent.ExecutionContext.Implicits.global)
    Await.ready(future, 1.hour)
  }

  def withAwaitResult[T](future: Future[T]): T = {
    Await.result(future, 1.hour)
  }

  def withAwaitResultAndClose[T](future: Future[T])(implicit es: ExecutionContextExecutorService): T = {
    future.onComplete { _ =>
      closeExecutionContext(es)
    }(scala.concurrent.ExecutionContext.Implicits.global)
    Await.result(future, 1.hour)
  }

  def initExecutionContext(numThreads: Int): ExecutionContextExecutorService = {
    val es = new ThreadPoolExecutor(numThreads, numThreads, 0L, MILLISECONDS, new LinkedBlockingQueue[Runnable],
      Executors.defaultThreadFactory, new ThreadPoolExecutor.AbortPolicy)
    ExecutionContext.fromExecutorService(es)
  }

  def closeExecutionContext(es: ExecutionContextExecutorService): Unit = {
    es.shutdown()
    if (!es.awaitTermination(1L, SECONDS)) {
      System.err.println("Error: ExecutorService does not exit itself, force to terminate.")
    }
  }
}
