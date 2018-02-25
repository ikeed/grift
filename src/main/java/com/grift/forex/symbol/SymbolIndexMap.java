package com.grift.forex.symbol;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import lombok.EqualsAndHashCode;
import org.jetbrains.annotations.NotNull;

@EqualsAndHashCode(callSuper = true)
public class SymbolIndexMap extends HashMap<String, Integer> {

    private static final long serialVersionUID = -6447422866018999830L;
    @NotNull
    private final transient Set<SymbolPair> allPairs;

    public SymbolIndexMap() {
        allPairs = new HashSet<>();
    }

    @NotNull
    @VisibleForTesting
    public SymbolIndexMap addSymbols(@NotNull List<String> symbols) {
        for (String sym : symbols) {
            addSymbol(sym);
        }
        return this;
    }

    @NotNull
    public SymbolIndexMap addSymbolPairs(@NotNull List<SymbolPair> pairs) {
        for (SymbolPair pair : pairs) {
            addSymbolPair(pair);
        }
        return this;
    }

    public final ImmutableSymbolIndexMap getImmutableCopy() {
        return new ImmutableSymbolIndexMap(this);
    }

    @NotNull
    public Integer[] addSymbolPair(@NotNull SymbolPair symbolPair) {
        addSymbol(symbolPair.getFirst());
        addSymbol(symbolPair.getSecond());
        allPairs.add(symbolPair);
        return getIndicesForSymbolPair(symbolPair);
    }

    public boolean addSymbol(String symbol) {
        if (containsKey(symbol)) {
            return false;
        }
        put(symbol, keySet().size());
        return true;
    }

    @NotNull
    private Integer getOrCreateSymbolIndex(String symbol) {
        addSymbol(symbol);
        return get(symbol);
    }

    @NotNull
    public Integer[] getIndicesForSymbolPair(@NotNull SymbolPair symbolPair) {
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

    public Set<SymbolPair> getAllPairs() {
        return ImmutableSet.copyOf(allPairs);
    }
}
