package com.grift.forex.symbol;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static lombok.Lombok.checkNotNull;

@Getter
public class SymbolPair {
    @NotNull
    private final String first;
    @NotNull
    private final String second;

    /**
     * Primary constructor.  Call it like new SymbolPair("CADUSD")
     *
     * @param pair symbol pair
     */
    public SymbolPair(@NotNull String pair) {
        this(checkNotNull(pair, "pair").substring(0, 3), pair.substring(3));
    }

    /**
     * Creates a new pair
     *
     * @param first  The first for this pair
     * @param second The second to use for this pair
     */
    public SymbolPair(@NotNull String first, @NotNull String second) {
        if (checkNotNull(first, "first").length() != 3 || checkNotNull(second, "second").length() != 3) {
            throw new IllegalArgumentException("must be two symbols of length 3.  Them's the rules.");
        } else if (isMalformedSymbol(first) || isMalformedSymbol(second)) {
            throw new IllegalArgumentException("Each symbol must be 3 letters e.g. \"USD\"");
        }

        this.first = first.toUpperCase();
        this.second = second.toUpperCase();
    }

    private static boolean isMalformedSymbol(@NotNull String first) {
        return !first.matches("[a-zA-Z]{3}");
    }

    @NotNull
    @Override
    public String toString() {
        return first + second;
    }

    @Override
    public boolean equals(@Nullable Object obj) {
        if (obj == null || !(obj instanceof String || obj instanceof SymbolPair)) {
            return false;
        }

        //works for String or SymbolPair
        //or really any object with toString defined the same as ours.
        return obj.toString().equals(toString());
    }

    @Override
    public int hashCode() {
        return toString().hashCode();
    }
}
