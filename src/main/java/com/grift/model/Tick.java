package com.grift.model;

import java.time.Instant;
import com.grift.forex.symbol.SymbolPair;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@AllArgsConstructor
@Builder
public class Tick {
    private SymbolPair symbolPair;
    private double val;
    private Instant timestamp;

    @Override
    public String toString() {
        return String.format("%s [%s]: %f", symbolPair, timestamp.toString(), val);
    }
}
