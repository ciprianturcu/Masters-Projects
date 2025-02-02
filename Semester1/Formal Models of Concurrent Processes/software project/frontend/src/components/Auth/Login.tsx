import React, { useState, useContext } from "react";
import AuthContext from "../../context/AuthContext";
import { useNavigate } from "react-router-dom";
import apiClient from "../../api/apiClient";

const Login: React.FC = () => {
  const [form, setForm] = useState({ username: "", password: "" });
  const [message, setMessage] = useState("");
  const auth = useContext(AuthContext);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await apiClient.post("/api/auth/login", form);
      localStorage.setItem("token", response.data.token)
      auth?.login(response.data.token);
      navigate("/transactions");
    } catch (error) {
      setMessage("Error: Invalid username or password.");
    }
  };

  return (
    <div>
        <form onSubmit={handleSubmit}>
      <h2>Login</h2>
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
      <button type="submit">Login</button>
      {message && <p>{message}</p>}
    </form>
    <button onClick={() => navigate("/register")}>Go to Register</button>
    </div>
  );
};

export default Login;
