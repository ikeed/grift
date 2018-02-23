package com.grift.forex.symbol;

import lombok.Getter;
import lombok.NonNull;
import org.jetbrains.annotations.NotNull;

import static lombok.Lombok.checkNotNull;

@Getter
public class SymbolPair {
    @NotNull
    @NonNull
    private final String first;
    @NotNull
    @NonNull
    private final String second;

    /**
     * Primary constructor.  Call it like new SymbolPair("CADUSD")
     *
     * @param pair symbol pair
     */
    public SymbolPair(@NotNull @NonNull String pair) {
        this(checkNotNull(pair, "pair").substring(0, 3), pair.substring(3));
    }

    @NotNull
    @Override
    public String toString() {
        return first + second;
    }

    /**
     * Creates a new pair
     *
     * @param first  The first for this pair
     * @param second The second to use for this pair
     */
    private SymbolPair(@NotNull @NonNull String first, @NotNull @NonNull String second) {
        if (checkNotNull(first, "first").length() != 3 || checkNotNull(second, "second").length() != 3) {
            throw new IllegalArgumentException("must be two symbols of length 3.  Them's the rules.");
        } else if (!isThreeLetters(first) || !isThreeLetters(second)) {
            throw new IllegalArgumentException("Each symbol must be 3 letters e.g. \"USD\"");
        }

        this.first = first.toUpperCase();
        this.second = second.toUpperCase();
    }

    private boolean isThreeLetters(@NonNull String first) {
        return first.matches("[a-zA-Z]{3}");
    }
}
