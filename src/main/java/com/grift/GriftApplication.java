package com.grift;

import java.util.List;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.decoupler.DecouplerMatrixColtImpl;
import com.grift.math.decoupler.Factory;
import com.grift.spring.service.DecoupleService;
import lombok.NonNull;
import org.jetbrains.annotations.NotNull;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class GriftApplication {
    @SuppressWarnings("SpellCheckingInspection")
    @NonNull
    @NotNull
    private final List<SymbolPair> symbolList = Lists.newArrayList(new SymbolPair("USDCAD"), new SymbolPair("CADGBP"));

    @NonNull
    @NotNull
    @Bean
    SymbolIndexMap symbolIndexMap() {
        return new SymbolIndexMap().addSymbolPairs(symbolList);
    }

    @NonNull
    @NotNull
    @Bean
    List<SymbolPair> symbolList() {
        return symbolList;
    }

    @NonNull
    @NotNull
    @Bean
    Factory decouplerFactory() {
        return new DecouplerMatrixColtImpl.ColtFactory(symbolIndexMap());
    }

    @NonNull
    @NotNull
    @Bean
    DecoupleService decouplerService() {
        return new DecoupleService(decouplerFactory());
    }

    public static void main(String[] args) {
        SpringApplication.run(GriftApplication.class, args);
    }
}
