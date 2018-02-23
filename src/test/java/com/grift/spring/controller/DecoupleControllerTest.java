package com.grift.spring.controller;

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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = {DecoupleController.class})
@ContextConfiguration(name = "fixture", classes = {GriftApplication.class})
public class DecoupleControllerTest {

    @Autowired
    DecoupleController decoupleController;

    @Test
    public void testPostTick() {
        Tick tickBody = Tick.builder()
                .symbolPair(new SymbolPair("CADUSD"))
                .timestamp(Instant.now())
                .val(12.12)
                .build();
        Tick tick = decoupleController.insertOrUpdateTick(tickBody);
        assertNotNull(tick);
        assertEquals("Symbol set", "CADUSD", tick.getSymbolPair().toString());
        assertNotNull("Time is set", tick.getTimestamp());
    }
}