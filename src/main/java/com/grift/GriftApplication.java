package com.grift;

import java.util.List;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.decoupler.DecouplerMatrixColtImpl;
import com.grift.math.decoupler.Factory;
import com.grift.spring.service.DecoupleService;
import org.jetbrains.annotations.NotNull;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class GriftApplication {
    @SuppressWarnings("SpellCheckingInspection")
    @NotNull
    private final List<SymbolPair> symbolList = Lists.newArrayList(new SymbolPair("USDCAD"), new SymbolPair("CADGBP"));

    @NotNull
    @Bean
    SymbolIndexMap symbolIndexMap() {
        return new SymbolIndexMap().addSymbolPairs(symbolList);
    }

    @NotNull
    @Bean
    List<SymbolPair> symbolList() {
        return symbolList;
    }

    @NotNull
    @Bean
    Factory decouplerFactory() {
        return new DecouplerMatrixColtImpl.ColtFactory(symbolIndexMap());
    }

    @NotNull
    @Bean
    DecoupleService decouplerService() {
        return new DecoupleService(decouplerFactory());
    }

    public static void main(String[] args) {
        SpringApplication.run(GriftApplication.class, args);
    }
}
