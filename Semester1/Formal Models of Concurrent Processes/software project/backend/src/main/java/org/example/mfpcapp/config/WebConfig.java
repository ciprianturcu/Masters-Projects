package org.example.mfpcapp.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

//    @Override
//    public void addCorsMappings(CorsRegistry registry) {
//        registry.addMapping("/**") // Allow all endpoints
//                .allowedOrigins("http://localhost:8080", "http://localhost:5137", "http://localhost:5137/") // Allow only your frontend origin
//                .allowedMethods("*") // Explicitly allow methods
//                .allowedHeaders("*") // Allow all headers
//                .allowCredentials(true); // Allow cookies and credentials (if needed)
//    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**") // Match all endpoints
                .allowedOrigins("*") // Allow all origins
                .allowedMethods("*") // Allow all HTTP methods
                .allowedHeaders("*") // Allow all headers
                .allowCredentials(false); // Disallow cookies if not needed
    }
}