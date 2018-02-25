package com.grift.math.decoupler;

import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import org.jetbrains.annotations.NotNull;

public interface DecouplerMatrix {
    @NotNull
    ProbabilityVector decouple();

    int rows();

    int columns();

    void put(@NotNull SymbolPair symbolPair, double val);

    double get(SymbolPair symbolPair);
}
