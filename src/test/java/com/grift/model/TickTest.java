package com.grift.model;

import java.time.Instant;
import com.grift.forex.symbol.SymbolPair;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TickTest {

    @Test
    public void testToString() {
        Tick tick = Tick.builder().symbolPair(new SymbolPair("CADUSD")).timestamp(Instant.ofEpochMilli(900000000)).val(10).build();
        assertEquals("toString mismatch", "CADUSD [1970-01-11T10:00:00Z]: 10.000000", tick.toString());
    }
}