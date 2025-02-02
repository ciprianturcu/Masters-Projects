package org.example.mfpcapp.service;

import org.example.mfpcapp.model.Transaction;
import org.example.mfpcapp.model.User;
import org.example.mfpcapp.repository.TransactionRepository;
import org.example.mfpcapp.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.LinkedHashMap;

@Service
public class TransactionService {

    @Autowired
    private TransactionRepository transactionRepository;

    private static final Logger logger = LoggerFactory.getLogger(TransactionService.class);

    @Autowired
    private UserRepository userRepository;

    /**
     * Retrieves all transactions for the currently authenticated user.
     */
    public List<Transaction> getAllTransactions() {
        User currentUser = getAuthenticatedUser();
        logger.info("current user: " + currentUser.getUsername());
        return transactionRepository.findByUser(currentUser);
    }

    /**
     * Retrieves a specific transaction by ID for the currently authenticated user.
     */
    public Transaction getTransactionById(Long id) {
        User currentUser = getAuthenticatedUser();
        return transactionRepository.findById(id)
                .filter(transaction -> transaction.getUser().getId().equals(currentUser.getId()))
                .orElse(null);
    }

    /**
     * Creates a new transaction for the currently authenticated user.
     */
    public Transaction createTransaction(Transaction transaction) {
        User currentUser = getAuthenticatedUser();
        transaction.setUser(currentUser); // Associate the transaction with the user
        return transactionRepository.save(transaction);
    }

    /**
     * Updates a transaction only if it belongs to the currently authenticated user.
     */
    public Transaction updateTransaction(Long id, Transaction updatedTransaction) {
        User currentUser = getAuthenticatedUser();
        return transactionRepository.findById(id)
                .filter(transaction -> transaction.getUser().getId().equals(currentUser.getId()))
                .map(transaction -> {
                    transaction.setDescription(updatedTransaction.getDescription());
                    transaction.setAmount(updatedTransaction.getAmount());
                    transaction.setCategory(updatedTransaction.getCategory());
                    transaction.setTransactionDate(updatedTransaction.getTransactionDate());
                    transaction.setUpdatedAt(updatedTransaction.getUpdatedAt());
                    return transactionRepository.save(transaction);
                })
                .orElse(null);
    }

    /**
     * Deletes a transaction only if it belongs to the currently authenticated user.
     */
    public void deleteTransaction(Long id) {
        User currentUser = getAuthenticatedUser();
        transactionRepository.findById(id)
                .filter(transaction -> transaction.getUser().getId().equals(currentUser.getId()))
                .ifPresent(transaction -> transactionRepository.deleteById(id));
    }

    /**
     * Helper method to retrieve the currently authenticated user.
     */
    private User getAuthenticatedUser() {
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        return userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("Authenticated user not found"));
    }

    public Map<String, Double> getTopSpendingCategories() {
        User currentUser = getAuthenticatedUser();
        List<Transaction> transactions = transactionRepository.findByUserUsername(currentUser.getUsername());

        // Group by category and sum amounts
        Map<String, Double> spendingByCategory = transactions.stream()
                .collect(Collectors.groupingBy(Transaction::getCategory, Collectors.summingDouble(Transaction::getAmount)));

        // Sort by highest spending and limit to top 3
        // Sort by highest spending and limit to top 3
        return spendingByCategory.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(3)
                .collect(Collectors.toMap(
                        Map.Entry::getKey, // Correctly extract key
                        Map.Entry::getValue, // Correctly extract value
                        (e1, e2) -> e1, // In case of conflict, use the first entry (not likely here)
                        LinkedHashMap::new // Maintain order of entries
                ));
    }
}
