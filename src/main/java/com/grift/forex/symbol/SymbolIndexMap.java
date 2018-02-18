package com.grift.forex.symbol;

import com.google.common.collect.Lists;
import lombok.NonNull;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

public class SymbolIndexMap extends HashMap<String, Integer> {

    public SymbolIndexMap addSymbols(@NonNull List<String> symbols) {
        for (String symbol : symbols) {
            addSymbol(symbol);
        }
        return this;
    }

    final public ImmutableSymbolIndexMap getImmutablecopy() {
        return new ImmutableSymbolIndexMap(this);
    }

    @NonNull
    public Integer[] addSymbolPair(@NonNull SymbolPair symbolPair) {
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

    @NonNull
    public Integer[] getIndecesForSymbolPair(SymbolPair symbolPair) {
        return new Integer[]{
                getOrCreateSymbolIndex(symbolPair.getFirst()),
                getOrCreateSymbolIndex(symbolPair.getSecond())
        };
    }

    public List<String> getAllSymbols() {
        List<String> arr = Lists.newArrayList(keySet());
        arr.sort(Comparator.comparing(this::get));
        return arr;
    }

    public int size() {
        return keySet().size();
    }
}
