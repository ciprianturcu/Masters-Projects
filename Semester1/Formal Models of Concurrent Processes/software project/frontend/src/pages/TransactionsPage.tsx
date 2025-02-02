import React, { useState, useEffect } from "react";
import TransactionForm from "../components/Transactions/TransactionForm";
import TransactionList from "../components/Transactions/TransactionList";
import apiClient from "../api/apiClient";
import { Transaction } from "../types";
import { useNavigate } from "react-router-dom";

const TransactionsPage: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [topCategories, setTopCategories] = useState<{ [key: string]: number }>({});
  const navigate = useNavigate();

  useEffect(() => {
    fetchTransactions();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token"); // Remove token from local storage
    navigate("/login"); // Redirect to login page
  };

  const fetchTopCategories = async () => {
    try {
      const response = await apiClient.get("/transactions/top-categories", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`, // Include token in header
        },
      });
      setTopCategories(response.data);
    } catch (error) {
      console.error("Error fetching top categories:", error);
    }
  };

  const sortByDate = () => {
    const sortedTransactions = [...transactions].sort((a, b) =>
      new Date(b.transactionDate).getTime() - new Date(a.transactionDate).getTime()
    );
    setTransactions(sortedTransactions);
  };

  const fetchTransactions = async () => {
    const response = await apiClient.get("/transactions");
    setTransactions(response.data);
  };

  const handleAddOrUpdate = async (transaction: Transaction) => {
    if (transaction.id) {
      await apiClient.put(`/transactions/${transaction.id}`, transaction);
    } else {
      await apiClient.post("/transactions", transaction);
    }
    fetchTransactions();
  };

  const handleDelete = async (id: number) => {
    await apiClient.delete(`/transactions/${id}`);
    fetchTransactions();
  };

  const handleClearSelection = () => {
    setSelectedTransaction(null); // Clear the selected transaction
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.heading}>Transactions</h1>
      <div style={styles.buttonContainer}>
        <button onClick={handleLogout} style={styles.logoutButton}>
          Logout
        </button>
        <button onClick={fetchTopCategories} style={styles.actionButton}>
          Top Spending Categories
        </button>
        <button onClick={sortByDate} style={styles.actionButton}>
          Sort by Date
        </button>
      </div>
      <div style={styles.content}>
        <TransactionForm
          selectedTransaction={selectedTransaction}
          onSubmit={handleAddOrUpdate}
          onClearSelection={handleClearSelection}
        />
        <TransactionList
          transactions={transactions}
          onSelect={setSelectedTransaction}
          onDelete={handleDelete}
        />
      </div>
      {Object.keys(topCategories).length > 0 && (
        <div style={styles.topCategories}>
          <h2>Top Spending Categories</h2>
          <ul>
            {Object.entries(topCategories).map(([category, total]) => (
              <li key={category}>
                {category}: ${total.toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

const styles = {
  page: {
    color: "#fff",
    minHeight: "100vh",
    minWidth: "100vh",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    padding: "20px",
  },
  heading: {
    marginBottom: "20px",
    fontSize: "2rem",
    fontWeight: "bold",
  },
  buttonContainer: {
    display: "flex",
    gap: "10px",
    marginBottom: "20px",
  },
  content: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "20px",
    width: "100%",
    maxWidth: "800px",
  },
  logoutButton: {
    padding: "10px 20px",
    backgroundColor: "#DC3545",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  actionButton: {
    padding: "10px 20px",
    backgroundColor: "#007BFF",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  topCategories: {
    marginTop: "20px",
    padding: "10px",
    backgroundColor: "#1a1a1a",
    borderRadius: "8px",
    width: "100%",
    maxWidth: "800px",
    textAlign: "center" as const,
  },
};
export default TransactionsPage;
