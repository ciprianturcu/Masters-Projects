import React, { useState } from "react";
import apiClient from "../../api/apiClient";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const Register: React.FC = () => {
  const [form, setForm] = useState({ username: "", password: "" });
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await apiClient.post("/api/auth/register", form); // Use apiClient here
      setMessage("Registration successful. Please log in.");
    } catch (error) {
      if (axios.isAxiosError(error)) {
        // Handle Axios error response
        setMessage("Error: " + (error.response?.data?.message || "An error occurred."));
      } else {
        // Handle other types of errors
        setMessage("An unexpected error occurred.");
      }
    }
  };

  return (
    <div>
        <form onSubmit={handleSubmit}> 
      <h2>Register</h2>
      <input
        type="text"
        placeholder="Username"
        value={form.username}
        onChange={(e) => setForm({ ...form, username: e.target.value })}
      />
      <input
        type="password"
        placeholder="Password"
        value={form.password}
        onChange={(e) => setForm({ ...form, password: e.target.value })}
      />
      <button type="submit">Register</button>
      {message && <p>{message}</p>}
    </form>
    <button onClick={() => navigate("/login")}>Go to Login</button>
    </div>
  );
};

export default Register;
