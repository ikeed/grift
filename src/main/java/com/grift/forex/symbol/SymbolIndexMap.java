package com.grift.forex.symbol;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import com.google.common.annotations.VisibleForTesting;
import lombok.EqualsAndHashCode;
import org.jetbrains.annotations.NotNull;

@EqualsAndHashCode(callSuper = true)
public class SymbolIndexMap extends HashMap<String, Integer> {

    private static final long serialVersionUID = -6447422866018999830L;

    @NotNull
    @VisibleForTesting
    public SymbolIndexMap addSymbols(@NotNull List<String> symbols) {
        symbols.forEach(this::addSymbol);
        return this;
    }

    @NotNull
    public SymbolIndexMap addSymbolPairs(@NotNull List<SymbolPair> pairs) {
        pairs.forEach(this::addSymbolPair);
        return this;
    }

    public final ImmutableSymbolIndexMap getImmutableCopy() {
        return new ImmutableSymbolIndexMap(this);
    }

    @NotNull
    public Integer[] addSymbolPair(@NotNull SymbolPair symbolPair) {
        addSymbol(symbolPair.getFirst());
        addSymbol(symbolPair.getSecond());
        return getIndicesForSymbolPair(symbolPair);
    }

    public boolean addSymbol(@NotNull String symbol) {
        if (containsKey(symbol)) {
            return false;
        }
        put(symbol, keySet().size());
        return true;
    }

    @NotNull
    private Integer getOrCreateSymbolIndex(@NotNull String symbol) {
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

}
