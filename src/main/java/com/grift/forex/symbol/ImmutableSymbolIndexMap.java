package com.grift.forex.symbol;

import lombok.NonNull;

import static com.google.common.base.Preconditions.checkNotNull;

public class ImmutableSymbolIndexMap extends SymbolIndexMap {
    public ImmutableSymbolIndexMap(@NonNull SymbolIndexMap symbolIndexMap) {
        putAll(checkNotNull(symbolIndexMap));
    }

    @Override
    public boolean addSymbol(String symbol) {
        if (containsKey(symbol)) {
            return true;
        }
        throw new UnsupportedOperationException("Immutable class is immutable");
    }

    @Override
    public Integer[] addSymbolPair(SymbolPair symbolPair) {
        if (containsKey(symbolPair.getFirst()) && containsKey(symbolPair.getSecond())) {
            return getIndecesForSymbolPair(symbolPair);
        }
        throw new UnsupportedOperationException("Immutable class is immutable");
    }
}
