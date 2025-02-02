import React, { useState, useEffect } from "react";
import { Transaction } from "../../types";

interface TransactionFormProps {
  selectedTransaction: Transaction | null;
  onSubmit: (transaction: Transaction) => void;
  onClearSelection: () => void;
}

const TransactionForm: React.FC<TransactionFormProps> = ({
  selectedTransaction,
  onSubmit,
  onClearSelection,
}) => {
  const [form, setForm] = useState<Transaction>({
    id: null,
    description: "",
    amount: 0,
    category: "",
    transactionDate: "",
  });

  const [validationErrors, setValidationErrors] = useState({
    description: "",
    amount: "",
    category: "",
    transactionDate: "",
  });

  useEffect(() => {
    if (selectedTransaction) {
      setForm(selectedTransaction);
    } else {
      setForm({ id: null, description: "", amount: 0, category: "", transactionDate: "" });
    }
  }, [selectedTransaction]);

  const validate = () => {
    const errors = {
      description: "",
      amount: "",
      category: "",
      transactionDate: "",
    };

    if (!form.description.trim()) {
      errors.description = "Description is required.";
    }

    if (form.amount <= 0) {
      errors.amount = "Amount must be greater than zero.";
    }

    if (!form.category.trim()) {
      errors.category = "Category is required.";
    }

    if (!form.transactionDate) {
      errors.transactionDate = "Transaction date is required.";
    }

    setValidationErrors(errors);

    // Check if there are any validation errors
    return Object.values(errors).every((error) => error === "");
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      onSubmit(form);
      clearForm();
    }
  };

  const clearForm = () => {
    setForm({ id: null, description: "", amount: 0, category: "", transactionDate: "" });
    setValidationErrors({
      description: "",
      amount: "",
      category: "",
      transactionDate: "",
    });
    onClearSelection();
  };

  const handleInputChange = (field: keyof Transaction, value: string | number) => {
    setForm({ ...form, [field]: value });
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <div>
        <input
          type="text"
          placeholder="Description"
          value={form.description}
          onChange={(e) => handleInputChange("description", e.target.value)}
          style={styles.input}
        />
        {validationErrors.description && <p style={styles.error}>{validationErrors.description}</p>}
      </div>
      <div>
        <input
          type="number"
          placeholder="Amount"
          value={form.amount}
          onChange={(e) => handleInputChange("amount", +e.target.value)}
          style={styles.input}
        />
        {validationErrors.amount && <p style={styles.error}>{validationErrors.amount}</p>}
      </div>
      <div>
        <input
          type="text"
          placeholder="Category"
          value={form.category}
          onChange={(e) => handleInputChange("category", e.target.value)}
          style={styles.input}
        />
        {validationErrors.category && <p style={styles.error}>{validationErrors.category}</p>}
      </div>
      <div>
        <input
          type="date"
          value={form.transactionDate}
          onChange={(e) => handleInputChange("transactionDate", e.target.value)}
          style={styles.input}
        />
        {validationErrors.transactionDate && (
          <p style={styles.error}>{validationErrors.transactionDate}</p>
        )}
      </div>
      <div style={styles.buttonContainer}>
        <button type="submit" style={styles.submitButton}>
          {form.id ? "Update" : "Add"} Transaction
        </button>
        <button type="button" onClick={clearForm} style={styles.clearButton}>
          Clear Fields
        </button>
      </div>
    </form>
  );
};

const styles = {
  form: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "10px",
    width: "100%",
    boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)",
    padding: "20px",
    borderRadius: "8px",
    backgroundColor: "#1a1a1a",
    color: "#fff",
  },
  input: {
    padding: "10px",
    border: "1px solid #ccc",
    borderRadius: "4px",
    backgroundColor: "#2a2a2a",
    color: "#fff",
    width: "100%", // Make input fill its container width
    boxSizing: "border-box", // Ensure padding is included in the total width
  },
  buttonContainer: {
    display: "flex",
    gap: "10px",
    justifyContent: "space-between",
  },
  error: {
    color: "#DC3545",
    fontSize: "0.9em",
    marginTop: "5px",
  },
  submitButton: {
    flex: 1,
    padding: "10px",
    border: "none",
    borderRadius: "4px",
    backgroundColor: "#007BFF",
    color: "#fff",
    cursor: "pointer",
    fontWeight: "bold",
    width: "100%", // Ensure button fills available space
    maxWidth: "300px", // Add a max width if necessary to prevent excessive stretching
  },
  clearButton: {
    flex: 1,
    padding: "10px",
    border: "none",
    borderRadius: "4px",
    backgroundColor: "#DC3545",
    color: "#fff",
    cursor: "pointer",
    fontWeight: "bold",
    width: "100%", // Ensure button fills available space
    maxWidth: "300px", // Add a max width if necessary
  },
};


export default TransactionForm;
