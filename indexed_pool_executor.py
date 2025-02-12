from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import _process_worker


class IndexedPoolExecutor(ProcessPoolExecutor):

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        super().__init__(max_workers, mp_context, initializer, initargs)

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
