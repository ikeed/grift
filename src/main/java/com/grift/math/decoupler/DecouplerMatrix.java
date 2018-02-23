package com.grift.math.decoupler;

import com.grift.math.ProbabilityVector;
import com.grift.forex.symbol.SymbolPair;
import lombok.NonNull;

public interface DecouplerMatrix {
    ProbabilityVector decouple();

    int rows();

    int columns();

    void put(@NonNull SymbolPair symbolPair, double val);

    double get(SymbolPair symbolPair);
}
