package com.grift.forex.symbol;


import com.google.common.collect.Lists;
import org.jetbrains.annotations.NotNull;

import static lombok.Lombok.checkNotNull;

public class ImmutableSymbolIndexMap extends SymbolIndexMap {
    private static final long serialVersionUID = -7832457250175407089L;

    public ImmutableSymbolIndexMap(@NotNull String... symbols) {
        this(new SymbolIndexMap().addSymbols(Lists.newArrayList(symbols)));
    }

    ImmutableSymbolIndexMap(@NotNull SymbolIndexMap symbolIndexMap) {
        putAll(checkNotNull(symbolIndexMap, "map"));
    }

    @Override
    public boolean addSymbol(@NotNull String symbol) {
        if (containsKey(symbol)) {
            return true;
        }
        throw new UnsupportedOperationException("Immutable class is immutable");
    }

    @NotNull
    @Override
    public Integer[] addSymbolPair(@NotNull SymbolPair symbolPair) {
        if (containsKey(symbolPair.getFirst()) && containsKey(symbolPair.getSecond())) {
            return getIndicesForSymbolPair(symbolPair);
        }
        throw new UnsupportedOperationException("Immutable class is immutable");
    }
}
