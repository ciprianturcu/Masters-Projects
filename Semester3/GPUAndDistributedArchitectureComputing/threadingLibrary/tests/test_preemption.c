#include "thread.h"
#include <stdio.h>

void *busy_thread(void *arg) {
    int id = *(int *)arg;
    printf("thread %d starting\n", id);

    for (int i = 0; i < 5; i++) {
        printf("[thread %d] working %d\n", id, i);
        for (volatile long j = 0; j < 200000000; j++);
    }
    
    printf("thread %d done\n", id);
    return NULL;
}

int main() {
    thread_type t1, t2;
    int id1 = 1, id2 = 2;
    
    printf("test for preemtive schedule test\n");
    printf("threads will NOT call yield - timer should preempt them\n\n");
    
    thread_create(&t1, busy_thread, &id1);
    thread_create(&t2, busy_thread, &id2);
    
    thread_join(t1, NULL);
    thread_join(t2, NULL);
    
    return 0;
}