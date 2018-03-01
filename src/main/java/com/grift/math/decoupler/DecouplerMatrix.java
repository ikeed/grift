package com.grift.math.decoupler;

import com.grift.forex.symbol.SymbolPair;
import com.grift.math.real.Real;
import com.grift.math.stats.ProbabilityVector;
import org.jetbrains.annotations.NotNull;

public interface DecouplerMatrix {
    @NotNull
    ProbabilityVector decouple();

    int rows();

    int columns();

    void put(@NotNull SymbolPair symbolPair, Real val);

    Real get(@NotNull SymbolPair symbolPair);

}
