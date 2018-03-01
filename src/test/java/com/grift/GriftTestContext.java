package com.grift;

import com.grift.spring.service.DecoupleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.stereotype.Component;

@TestConfiguration
@Component
public class GriftTestContext {
    @Autowired
    DecoupleService decoupleService;

//    @Bean
//    MarketSimulator marketSimulator() {
//        return new MarketSimulator(decoupleService)
//    }
}
