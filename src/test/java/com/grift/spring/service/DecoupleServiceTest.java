package com.grift.spring.service;

import java.time.Instant;
import com.grift.GriftApplication;
import com.grift.forex.symbol.SymbolPair;
import com.grift.model.Tick;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = DecoupleService.class)
@ContextConfiguration(classes = GriftApplication.class)
public class DecoupleServiceTest {

    @Autowired
    DecoupleService decoupleService;

    @Test
    public void insertTick() {
        decoupleService.insertTick(new Tick(new SymbolPair("USDCAD"), 10.12, Instant.now()));
    }
}