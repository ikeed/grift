package com.grift.math.decoupler;

import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.ProbabilityVector;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import lombok.NonNull;

import static com.google.common.base.Preconditions.checkNotNull;

interface DecouplerMatrix {
    ProbabilityVector decouple();

    int rows();

    int columns();

    void put(@NonNull SymbolPair symbolPair, double val);

    double get(SymbolPair symbolPair);

    class Factory {
        @NonNull
        private final ImmutableSymbolIndexMap symbolIndexMap;

        public Factory(@NonNull SymbolIndexMap symbolIndexMap) {
            this.symbolIndexMap = new ImmutableSymbolIndexMap(checkNotNull(symbolIndexMap));
        }

        public DecouplerMatrixColtImpl make(@NonNull double[][] doubles) {
            if (doubles.length != symbolIndexMap.size() || doubles[0].length != symbolIndexMap.size()) {
                throw new IllegalArgumentException("The array of values doesn't match the number of symbols in the map");
            }
            return new DecouplerMatrixColtImpl(symbolIndexMap, doubles);
        }

        public DecouplerMatrixColtImpl make() {
            return new DecouplerMatrixColtImpl(symbolIndexMap);
        }
    }
}
