package org.example.mfpcapp.security;

import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Component;

import java.security.Key;
import java.util.Date;

@Component
public class JwtUtil {

    // Define your secret key. It must be at least 32 bytes long for HS256.
    private static final String SECRET_KEY = "yourStrongSecretKeyThatIsAtLeast32BytesLong!";
    private static final long EXPIRATION_TIME_MS = 86400000L; // Token validity (24 hours in milliseconds)

    // Generate the signing key
    private final Key signingKey = Keys.hmacShaKeyFor(SECRET_KEY.getBytes());

    /**
     * Generate a JWT token for a given username.
     *
     * @param username the username to include in the token
     * @return a signed JWT token
     */
    public String generateToken(String username) {
        return Jwts.builder()
                .setSubject(username) // Set the username as the subject
                .setIssuedAt(new Date()) // Set the token issue time
                .setExpiration(new Date(System.currentTimeMillis() + EXPIRATION_TIME_MS)) // Set expiration time
                .signWith(signingKey, SignatureAlgorithm.HS256) // Sign the token with HS256 and the key
                .compact(); // Generate the compact token
    }

    /**
     * Validate a given JWT token.
     *
     * @param token the JWT token to validate
     * @return true if the token is valid, false otherwise
     */
    public boolean validateToken(String token) {
        try {
            // Parse the token to ensure it is valid
            Jwts.parserBuilder()
                    .setSigningKey(signingKey) // Set the signing key
                    .build()
                    .parseClaimsJws(token); // Parse the claims
            return true; // If no exceptions, the token is valid
        } catch (JwtException | IllegalArgumentException e) {
            // Token is invalid (expired, malformed, etc.)
            System.out.println("Invalid JWT Token: " + e.getMessage());
            return false;
        }
    }

    /**
     * Extract the username (subject) from a given JWT token.
     *
     * @param token the JWT token
     * @return the username included in the token
     */
    public String extractUsername(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(signingKey) // Set the signing key
                .build()
                .parseClaimsJws(token) // Parse the claims
                .getBody()
                .getSubject(); // Extract the subject (username)
    }

    /**
     * Extract all claims from a JWT token.
     *
     * @param token the JWT token
     * @return the claims included in the token
     */
    public Claims extractAllClaims(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(signingKey) // Set the signing key
                .build()
                .parseClaimsJws(token) // Parse the claims
                .getBody(); // Get the claims body
    }
}