package org.example.mfpcapp.controller;


import org.example.mfpcapp.dto.JwtResponse;
import org.example.mfpcapp.dto.LoginRequest;
import org.example.mfpcapp.dto.RegisterRequest;
import org.example.mfpcapp.security.JwtUtil;
import org.example.mfpcapp.service.UserService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @Autowired
    private AuthenticationManager authenticationManager;

    private static final Logger logger = LoggerFactory.getLogger(AuthController.class);

    @Autowired
    private UserService userService;

    @Autowired
    private JwtUtil jwtUtil;

    @PostMapping("/register")
    public ResponseEntity<?> registerUser(@RequestBody RegisterRequest registerRequest) {
        if (userService.existsByUsername(registerRequest.getUsername())) {
            return ResponseEntity
                    .badRequest()
                    .body("Error: Username is already taken!");
        }

        userService.registerUser(registerRequest.getUsername(), registerRequest.getPassword());

        return ResponseEntity.ok("User registered successfully!");
    }

    @PostMapping("/login")
    public ResponseEntity<?> authenticateUser(@RequestBody LoginRequest loginRequest) {
        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(loginRequest.getUsername(), loginRequest.getPassword())
            );
            String jwt = jwtUtil.generateToken(authentication.getName());
            return ResponseEntity.ok(new JwtResponse(jwt));
        } catch (AuthenticationException ex) {
            return ResponseEntity
                    .status(401)
                    .body("Error: Invalid username or password.");
        } catch (Exception ex) {
            return ResponseEntity
                    .status(500)
                    .body("Error: An unexpected error occurred. Please try again.");
        }
    }
}