package com.grift.forex.symbol;

import lombok.NonNull;
import org.jetbrains.annotations.NotNull;

import static lombok.Lombok.checkNotNull;

public class ImmutableSymbolIndexMap extends SymbolIndexMap {
    public ImmutableSymbolIndexMap(@NotNull @NonNull SymbolIndexMap symbolIndexMap) {
        putAll(checkNotNull(symbolIndexMap, "map"));
    }

    @Override
    public boolean addSymbol(String symbol) {
        if (containsKey(symbol)) {
            return true;
        }
        throw new UnsupportedOperationException("Immutable class is immutable");
    }

    @NotNull
    @Override
    public Integer[] addSymbolPair(@NotNull SymbolPair symbolPair) {
        if (containsKey(symbolPair.getFirst()) && containsKey(symbolPair.getSecond())) {
            return getIndecesForSymbolPair(symbolPair);
        }
        throw new UnsupportedOperationException("Immutable class is immutable");
    }
}
