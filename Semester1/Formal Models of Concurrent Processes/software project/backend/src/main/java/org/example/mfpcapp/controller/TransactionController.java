package org.example.mfpcapp.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.example.mfpcapp.model.Transaction;
import org.example.mfpcapp.service.TransactionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/transactions")
@Tag(name = "Transaction API", description = "Manage Transactions")
public class TransactionController {

    @Autowired
    private TransactionService transactionService;

    @Operation(summary = "Get all transactions")
    @GetMapping
    public List<Transaction> getAllTransactions() {
        return transactionService.getAllTransactions();
    }

    @Operation(summary = "Get a transaction by ID")
    @GetMapping("/{id}")
    public Transaction getTransactionById(@PathVariable Long id) {
        return transactionService.getTransactionById(id);
    }

    @Operation(summary = "Create a new transaction")
    @PostMapping
    public Transaction createTransaction(@RequestBody Transaction transaction) {
        return transactionService.createTransaction(transaction);
    }

    @Operation(summary = "Update a transaction by ID")
    @PutMapping("/{id}")
    public Transaction updateTransaction(@PathVariable Long id, @RequestBody Transaction transaction) {
        return transactionService.updateTransaction(id, transaction);
    }

    @Operation(summary = "Delete a transaction by ID")
    @DeleteMapping("/{id}")
    public void deleteTransaction(@PathVariable Long id) {
        transactionService.deleteTransaction(id);
    }

    @GetMapping("/top-categories")
    public ResponseEntity<?> getTopSpendingCategories() {
        Map<String, Double> topCategories = transactionService.getTopSpendingCategories();

        return ResponseEntity.ok(topCategories);
    }
}