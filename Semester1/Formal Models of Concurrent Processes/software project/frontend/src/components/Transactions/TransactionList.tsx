import React from "react";
import { Transaction } from "../../types";

interface TransactionListProps {
  transactions: Transaction[];
  onSelect: (transaction: Transaction) => void;
  onDelete: (id: number) => void;
}

const TransactionList: React.FC<TransactionListProps> = ({ transactions, onSelect, onDelete }) => {
    return (
        <table style={styles.table}>
          <thead>
            <tr>
              <th>Description</th>
              <th>Amount</th>
              <th>Category</th>
              <th>Date</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((transaction) => (
              <tr key={transaction.id}>
                <td>{transaction.description}</td>
                <td>{transaction.amount}</td>
                <td>{transaction.category}</td>
                <td>{transaction.transactionDate}</td>
                <td>
                  <button
                    onClick={() => onSelect(transaction)}
                    style={styles.button}
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => onDelete(transaction.id!)}
                    style={{ ...styles.button, backgroundColor: "#DC3545" }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      );
};

const styles = {
    table: {
      width: "100%",
      borderCollapse: "collapse" as const,
      boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)",
      borderRadius: "8px",
      overflow: "hidden",
    },
    row: {
      backgroundColor: "#f9f9f9",
      textAlign: "left",
      borderBottom: "1px solid #ddd",
    },
    button: {
      padding: "5px 10px",
      borderRadius: "4px",
      border: "none",
      backgroundColor: "#007BFF",
      color: "#fff",
      cursor: "pointer",
      fontWeight: "bold",
      margin: "0 5px",
    },
  };

export default TransactionList;
