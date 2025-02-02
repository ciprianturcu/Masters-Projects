import React from "react";

const SimplePage: React.FC = () => {
  return (
    <div>
      <h1>Welcome to the Simple Page</h1>
      <p>This page is accessible without authentication or the AuthProvider context.</p>
    </div>
  );
};

export default SimplePage;