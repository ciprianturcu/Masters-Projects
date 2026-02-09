#include "thread.h"
#include "mutex.h"
#include "condvar.h"
#include <stdio.h>

#define NUM_ACCOUNTS 3
#define NUM_CUSTOMERS 4
#define TRANSACTIONS_PER_CUSTOMER 5

typedef struct {
    int balance;
    mutex_type *lock;
    condvar_type *balance_available;
} account_t;

account_t accounts[NUM_ACCOUNTS];

void init_accounts(void) {
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        accounts[i].balance = 1000;
        mutex_init(&accounts[i].lock);
        cond_init(&accounts[i].balance_available);
    }
}

void cleanup_accounts(void) {
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        cond_destroy(accounts[i].balance_available);
        mutex_destroy(accounts[i].lock);
    }
}

void transfer(int from, int to, int amount, int customer_id) {
    int first = (from < to) ? from : to;
    int second = (from < to) ? to : from;
    
    printf("[customer %d] initiating transfer: $%d from account %d to account %d\n",
           customer_id, amount, from, to);

    mutex_lock(accounts[first].lock);
    printf("[customer %d] locked Account %d\n", customer_id, first);

    mutex_lock(accounts[second].lock);
    printf("[customer %d] locked Account %d\n", customer_id, second);

    while (accounts[from].balance < amount) {
        printf("[customer %d] insufficient funds in account %d ($%d < $%d), waiting\n",
               customer_id, from, accounts[from].balance, amount);
        cond_wait(accounts[from].balance_available, accounts[from].lock);
    }

    accounts[from].balance -= amount;
    accounts[to].balance += amount;

    printf("[customer %d] ✓ transfer complete: account %d ($%d) → account %d ($%d)\n",
           customer_id, from, accounts[from].balance, to, accounts[to].balance);

    cond_signal(accounts[to].balance_available);

    mutex_unlock(accounts[second].lock);
    mutex_unlock(accounts[first].lock);
}

void *customer_thread(void *arg) {
    int id = *(int *)arg;
    
    printf("[customer %d] starting transactions\n", id);

    for (int i = 0; i < TRANSACTIONS_PER_CUSTOMER; i++) {
        int from = (id + i) % NUM_ACCOUNTS;
        int to = (id + i + 1) % NUM_ACCOUNTS;
        int amount = 50 + (id * 10) + (i * 5);
        
        transfer(from, to, amount, id);

        thread_yield();
        for (volatile long j = 0; j < 50000000; j++);
    }
    
    printf("[customer %d] all transactions complete\n", id);
    return NULL;
}

void *auditor_thread(void *arg) {
    (void)arg;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < NUM_ACCOUNTS; j++) {
            mutex_lock(accounts[j].lock);
        }

        int total = 0;
        printf("\n[auditor] account balances: ");
        for (int j = 0; j < NUM_ACCOUNTS; j++) {
            printf("acc%d=$%d ", j, accounts[j].balance);
            total += accounts[j].balance;
        }
        printf("total=$%d\n\n", total);

        for (int j = 0; j < NUM_ACCOUNTS; j++) {
            mutex_unlock(accounts[j].lock);
        }

        for (volatile long j = 0; j < 200000000; j++);
    }
    
    return NULL;
}

int main() {
    printf("bank sys test (NO DEADLOCK)\n");
    printf("features tested: threads, mutexes condvar\n");
    
    init_accounts();
    
    thread_type customers[NUM_CUSTOMERS];
    thread_type auditor;
    int ids[NUM_CUSTOMERS];

    for (int i = 0; i < NUM_CUSTOMERS; i++) {
        ids[i] = i + 1;
        thread_create(&customers[i], customer_thread, &ids[i]);
    }

    thread_create(&auditor, auditor_thread, NULL);

    for (int i = 0; i < NUM_CUSTOMERS; i++) {
        thread_join(customers[i], NULL);
    }

    thread_join(auditor, NULL);

    printf("final account balances:\n");
    int total = 0;
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        printf("  account %d: $%d\n", i, accounts[i].balance);
        total += accounts[i].balance;
    }
    printf("  total: $%d (should be $%d)\n", total, NUM_ACCOUNTS * 1000);
    
    cleanup_accounts();
    
    return 0;
}