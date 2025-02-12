from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import _process_worker


class IndexedProcessPoolExecutor(ProcessPoolExecutor):

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        super().__init__(max_workers, mp_context, initializer, initargs)

    def _adjust_process_count(self):
        # if there's an idle process, we don't need to spawn a new one.
        if self._idle_worker_semaphore.acquire(blocking=False):
            return

        process_count = len(self._processes)
        if process_count < self._max_workers:
            # Assertion disabled as this codepath is also used to replace a
            # worker that unexpectedly dies, even when using the 'fork' start
            # method. That means there is still a potential deadlock bug. If a
            # 'fork' mp_context worker dies, we'll be forking a new one when
            # we know a thread is running (self._executor_manager_thread).
            #assert self._safe_to_dynamically_spawn_children or not self._executor_manager_thread, 'https://github.com/python/cpython/issues/90622'
            self._spawn_process()

    def _launch_processes(self):
        # https://github.com/python/cpython/issues/90622
        assert not self._executor_manager_thread, (
                'Processes cannot be fork()ed after the thread has started, '
                'deadlock in the child processes could result.')
        for i in range(len(self._processes), self._max_workers):
            self._spawn_process(i)

    def _spawn_process(self, i: int):
        p = self._mp_context.Process(
            target=_process_worker,
            args=(self._call_queue,
                  self._result_queue,
                  self._initializer,
                  (i, *self._initargs)))
        p.start()
        self._processes[p.pid] = p
