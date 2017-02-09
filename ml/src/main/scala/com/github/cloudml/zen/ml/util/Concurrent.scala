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
  import SimpleConcurrentBackend._

  type ThID = Int

  case class ParaExecutionContext(nThreads: Int, es: ExecutionContextExecutorService)

  def newParaExecutionContext(nThreads: Int): ParaExecutionContext = {
    ParaExecutionContext(nThreads, initExecutionContext(nThreads))
  }

  // coarse-grained multi-threading, simple
  def parallelized_foreachSplit(totalSize: Int,
    funcThrd: (Int, Int, ThID) => Unit,
    closing: Boolean = false)(implicit pec: ParaExecutionContext): Unit = {
    val nThreads = pec.nThreads
    implicit val ec = pec.es
    val sizePerThrd = {
      val npt = totalSize / nThreads
      if (npt * nThreads == totalSize) npt else npt + 1
    }
    val all = for (thid <- 0 until nThreads) yield withFuture {
      val ss = sizePerThrd * thid
      val sn = math.min(ss + sizePerThrd, totalSize)
      funcThrd(ss, sn, thid)
    }
    val bf = if (closing) withAwaitReadyAndClose[IndexedSeq[Unit]] _ else withAwaitReady[IndexedSeq[Unit]] _
    bf(Future.sequence(all))
  }

  // work-stealing mode multi-threading, more load-balanced
  def parallelized_foreachElement[T](totalIter: Iterator[T],
    funcThrd: T => Unit,
    closing: Boolean = false)(implicit pec: ParaExecutionContext): Unit = {
    implicit val ec = pec.es
    val all = for (e <- totalIter) yield withFuture(funcThrd(e))
    val bf = if (closing) withAwaitReadyAndClose[Iterator[Unit]] _ else withAwaitReady[Iterator[Unit]] _
    bf(Future.sequence(all))
  }

  // mini-batch between coarse-grained & work-stealing, most efficient
  def parallelized_foreachBatch[T](totalIter: Iterator[T],
    nBatch: Int,
    funcThrd: (Seq[T], ThID) => Unit,
    closing: Boolean = false)(implicit pec: ParaExecutionContext): Unit = {
    implicit val ec = pec.es
    val thq = new ConcurrentLinkedQueue(1 to pec.nThreads)
    val all = totalIter.grouped(nBatch).map(batch => withFuture {
      val thid = thq.poll() - 1
      try {
        funcThrd(batch, thid)
      } finally {
        thq.add(thid + 1)
      }
    })
    val bf = if (closing) withAwaitReadyAndClose[Iterator[Unit]] _ else withAwaitReady[Iterator[Unit]] _
    bf(Future.sequence(all))
  }

  def parallelized_mapElement[T, U](totalIter: Iterator[T],
    funcThrd: T => U,
    closing: Boolean = false)(implicit pec: ParaExecutionContext): Iterator[U] = {
    implicit val ec = pec.es
    val all = for (e <- totalIter) yield withFuture(funcThrd(e))
    val bf = if (closing) withAwaitResultAndClose[Iterator[U]] _ else withAwaitResult[Iterator[U]] _
    bf(Future.sequence(all))
  }

  def parallelized_mapBatch[T, U](totalIter: Iterator[T],
    funcThrd: T => U,
    closing: Boolean = false)(implicit pec: ParaExecutionContext): Iterator[U] = {
    implicit val ec = pec.es
    totalIter.grouped(pec.nThreads).flatMap { batch =>
      val all = Future.traverse(batch)(e => withFuture(funcThrd(e)))
      withAwaitResult(all)
    } ++ {
      if (closing) {
        closeExecutionContext(ec)
      }
      Iterator.empty
    }
  }

  def parallelized_reduceSplit[U](totalSize: Int,
    funcThrd: (Int, Int, ThID) => U,
    reducer: (U, U) => U,
    closing: Boolean = false)(implicit pec: ParaExecutionContext): U = {
    val nThreads = pec.nThreads
    implicit val ec = pec.es
    val sizePerThrd = {
      val npt = totalSize / nThreads
      if (npt * nThreads == totalSize) npt else npt + 1
    }
    val all = for (thid <- 0 until nThreads) yield withFuture {
      val ss = sizePerThrd * thid
      val sn = math.min(ss + sizePerThrd, totalSize)
      funcThrd(ss, sn, thid)
    }
    val bf = if (closing) withAwaitResultAndClose[U] _ else withAwaitResult[U] _
    bf(Future.reduce(all)(reducer))
  }

  def parallelized_reduceBatch[T, U](totalIter: Iterator[T],
    nBatch: Int,
    funcThrd: (Seq[T], ThID) => U,
    reducer: (U, U) => U,
    closing: Boolean = false)(implicit pec: ParaExecutionContext): U = {
    implicit val ec = pec.es
    val thq = new ConcurrentLinkedQueue(1 to pec.nThreads)
    val all = totalIter.grouped(nBatch).map(batch => withFuture {
      val thid = thq.poll() - 1
      try {
        funcThrd(batch, thid)
      } finally {
        thq.add(thid + 1)
      }
    })
    val bf = if (closing) withAwaitResultAndClose[U] _ else withAwaitResult[U] _
    bf(Future.reduce(all)(reducer))
  }
}

object SimpleConcurrentBackend extends Serializable {
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

  @inline def initExecutionContext(nThreads: Int): ExecutionContextExecutorService = {
    ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(nThreads))
  }

  @inline def closeExecutionContext(es: ExecutionContextExecutorService): Unit = {
    es.shutdown()
  }
}

object DebugConcurrentBackend extends Serializable {
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

  def initExecutionContext(nThreads: Int): ExecutionContextExecutorService = {
    val es = new ThreadPoolExecutor(nThreads, nThreads, 0L, MILLISECONDS, new LinkedBlockingQueue[Runnable],
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
