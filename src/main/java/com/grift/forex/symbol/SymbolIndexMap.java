package com.grift.forex.symbol;

import lombok.NonNull;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

public class SymbolIndexMap extends HashMap<String, Integer> {

    @NotNull
    public SymbolIndexMap addSymbols(@NotNull @NonNull List<String> symbols) {
        for (String symbol : symbols) {
            addSymbol(symbol);
        }
        return this;
    }

    final public ImmutableSymbolIndexMap getImmutablecopy() {
        return new ImmutableSymbolIndexMap(this);
    }

    @NonNull
    public Integer[] addSymbolPair(@NotNull @NonNull SymbolPair symbolPair) {
        addSymbol(symbolPair.getFirst());
        addSymbol(symbolPair.getSecond());
        return getIndecesForSymbolPair(symbolPair);
    }

    public boolean addSymbol(String symbol) {
        if (containsKey(symbol)) {
            return false;
        }
        put(symbol, keySet().size());
        return true;
    }

    @NonNull
    private Integer getOrCreateSymbolIndex(String symbol) {
        addSymbol(symbol);
        return get(symbol);
    }

    @NotNull
    @NonNull
    public Integer[] getIndecesForSymbolPair(@NotNull SymbolPair symbolPair) {
        return new Integer[]{
                getOrCreateSymbolIndex(symbolPair.getFirst()),
                getOrCreateSymbolIndex(symbolPair.getSecond())
        };
    }

    @NotNull
    public List<String> getAllSymbols() {
        List<String> arr = new ArrayList<>(keySet());
        arr.sort(Comparator.comparing(this::get));
        return arr;
    }

    public int size() {
        return keySet().size();
    }
}
