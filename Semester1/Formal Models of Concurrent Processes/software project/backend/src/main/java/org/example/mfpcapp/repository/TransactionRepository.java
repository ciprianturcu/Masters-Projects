package org.example.mfpcapp.repository;

import org.example.mfpcapp.model.Transaction;
import org.example.mfpcapp.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface TransactionRepository extends JpaRepository<Transaction, Long> {
    List<Transaction> findByUser(User currentUser);

    List<Transaction> findByUserUsername(String username);
}
