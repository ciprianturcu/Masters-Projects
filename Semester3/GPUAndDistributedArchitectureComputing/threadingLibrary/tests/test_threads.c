#include "thread.h"
#include <stdio.h>

void *worker(void *arg) {
    int id = *(int *)arg;
    
    for (int i = 0; i < 5; i++) {
        printf("thread %d: iteration %d\n", id, i);
        thread_yield();
    }
    
    printf("thread %d: exiting\n", id);
    return (void *)(long)(id * 100);
}

int main() {
    thread_type t1, t2, t3;
    int id1 = 1, id2 = 2, id3 = 3;
    
    printf("main: creating threads\n");
    
    thread_create(&t1, worker, &id1);
    thread_create(&t2, worker, &id2);
    thread_create(&t3, worker, &id3);
    
    printf("main: waiting for threads\n");
    
    void *ret1, *ret2, *ret3;
    thread_join(t1, &ret1);
    thread_join(t2, &ret2);
    thread_join(t3, &ret3);
    
    printf("main: threads finished with return values: %ld, %ld, %ld\n",
           (long)ret1, (long)ret2, (long)ret3);
    
    return 0;
}